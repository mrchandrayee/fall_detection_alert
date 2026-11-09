from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import json, math, time
import cv2
import numpy as np

# COCO-17 keypoint indices: nose, eyes, ears, shoulders, elbows, wrists,
# hips, knees, ankles.
SKELETON_CONNECTIONS=[
    (5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
    (11,12),(11,13),(13,15),(12,14),(14,16),(0,1),(0,2),
    (1,3),(2,4),(3,5),(4,6)
]
COLOR_FACE=(255,200,100)
COLOR_TORSO=(255,150,50)
COLOR_ARMS=(255,158,50)
COLOR_LEGS=(200,169,255)
STATUS_COLORS={
    "STANDING":(0,228,90),
    "FALLING":(0,165,255),
    "FALLEN":(0,0,255),
    "RECOVERING":(255,190,0),
}

class FallKinematicAnalyzer:
    """Estimate body orientation from COCO-17 pose landmarks and bounding box."""
    def __init__(self,keypoint_confidence=0.30):
        self.keypoint_confidence=float(keypoint_confidence)

    @staticmethod
    def _midpoint(points, confidences, indices, threshold):
        valid=[i for i in indices if i<len(points) and confidences[i]>=threshold]
        if not valid: return None
        return np.mean(points[valid],axis=0)

    def analyze_pose(self,keypoints_xy,keypoints_conf,box):
        x1,y1,x2,y2=box
        bw=max(1,x2-x1); bh=max(1,y2-y1)
        aspect_ratio=bw/bh
        torso_angle=None
        head_hip_ratio=None
        has_pose=False
        if keypoints_xy is not None and len(keypoints_xy)>=17:
            pts=np.asarray(keypoints_xy,dtype=float)
            conf=np.asarray(keypoints_conf,dtype=float)
            sh=self._midpoint(pts,conf,[5,6],self.keypoint_confidence)
            hip=self._midpoint(pts,conf,[11,12],self.keypoint_confidence)
            if sh is not None and hip is not None:
                dx=float(sh[0]-hip[0]); dy=float(sh[1]-hip[1])
                torso_angle=math.degrees(math.atan2(abs(dy),abs(dx)+1e-8))
                has_pose=True
                nose_conf=conf[0] if len(conf)>0 else 0
                if nose_conf>=self.keypoint_confidence:
                    nose_y=float(pts[0][1])
                    head_hip_ratio=float((hip[1]-nose_y)/bh)
        return {
            "has_pose":has_pose,
            "torso_angle":torso_angle,
            "aspect_ratio":aspect_ratio,
            "head_hip_ratio":head_hip_ratio,
            "box_width":bw,
            "box_height":bh,
        }

class PersonFallTracker:
    """Temporal state machine with posture, rapid-drop, and recovery hysteresis."""
    def __init__(self,track_id,history_len=15,fps=25.0):
        self.track_id=track_id
        self.state="STANDING"
        self.fallen_frames=0
        self.recover_frames=0
        self.history=deque(maxlen=history_len)
        self.last_update_time=time.time()
        self.fps=max(1.0,float(fps))
        self.missed_frames=0
        self.event_started=None

    def mark_missed(self):
        self.missed_frames+=1

    def update(self,box,pose_metrics):
        x1,y1,x2,y2=box
        cx,cy=(x1+x2)/2.0,(y1+y2)/2.0
        now=time.time()
        dt=max(1.0/self.fps,now-self.last_update_time)
        self.last_update_time=now
        self.missed_frames=0
        angle=pose_metrics.get("torso_angle")
        ar=float(pose_metrics.get("aspect_ratio",0))
        hr=pose_metrics.get("head_hip_ratio")

        vy=0.0
        if len(self.history)>=3:
            prev_t,prev_x,prev_y,*_=self.history[-3]
            dt_v=max(0.05,now-prev_t)
            vy=(cy-prev_y)/dt_v  # Positive means downward movement in image coordinates.
        self.history.append((now,cx,cy,angle,ar))

        # Decide whether the body is horizontal/lying.
        is_horizontal=False
        if angle is not None:
            if angle<45.0:
                is_horizontal=True
            elif angle<55.0 and ar>0.55:
                is_horizontal=True
            elif angle<65.0 and ar>0.65:
                is_horizontal=True
        elif ar>0.85:
            is_horizontal=True
        if hr is not None and hr<0.30:
            is_horizontal=True

        is_upright=False
        if angle is not None and angle>=65.0 and ar<0.75:
            is_upright=True
        elif angle is None and ar<0.50:
            is_upright=True

        fast_drop=False
        if not is_upright and len(self.history)>=4:
            fast_drop=vy>max(180.0,7.0*self.fps)

        if self.state=="STANDING":
            if fast_drop:
                self.state="FALLING"
                self.fallen_frames=0
                self.event_started=now
            elif is_horizontal:
                self.fallen_frames+=1
                if self.fallen_frames>=3:
                    self.state="FALLEN"
                    self.event_started=now
            elif is_upright:
                self.fallen_frames=max(0,self.fallen_frames-1)
        elif self.state=="FALLING":
            if is_horizontal:
                self.state="FALLEN"
                self.fallen_frames=10
            elif is_upright:
                self.state="STANDING"
                self.fallen_frames=0
            else:
                self.fallen_frames+=1
                if self.fallen_frames>=max(4,int(self.fps*0.5)):
                    self.state="FALLEN"
        elif self.state=="FALLEN":
            if is_upright:
                self.state="RECOVERING"
                self.recover_frames=1
        elif self.state=="RECOVERING":
            if is_horizontal:
                self.state="FALLEN"
                self.recover_frames=0
            elif is_upright:
                self.recover_frames+=1
                if self.recover_frames>=max(10,int(self.fps*0.4)):
                    self.state="STANDING"
                    self.fallen_frames=0
                    self.recover_frames=0
            # Ambiguous pose: hold RECOVERING instead of rapidly changing state.
        return self.state

class FallEventLogger:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=self.path.open("a",encoding="utf-8")
        self.logged=set()
        self.event_count=0

    def log(self,track_id,frame_idx,fps,metrics):
        # Log one event per track ID for this processing run.
        if track_id in self.logged: return
        self.logged.add(track_id); self.event_count+=1
        row={
            "event":"FALL_DETECTED","track_id":int(track_id),
            "frame":int(frame_idx),"timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "video_time_seconds":round(frame_idx/max(1.0,fps),2),
            "metrics":metrics
        }
        self.file.write(json.dumps(row)+"\n"); self.file.flush()

    def close(self):
        self.file.close()

def draw_skeleton(frame,keypoints,confidences,threshold=0.30):
    pts=np.asarray(keypoints)
    conf=np.asarray(confidences)
    for a,b in SKELETON_CONNECTIONS:
        if a<len(pts) and b<len(pts) and conf[a]>=threshold and conf[b]>=threshold:
            pa=tuple(np.round(pts[a]).astype(int)); pb=tuple(np.round(pts[b]).astype(int))
            color=COLOR_ARMS if a in (5,6,7,8,9,10) else COLOR_LEGS
            cv2.line(frame,pa,pb,color,2,cv2.LINE_AA)
    for i,pt in enumerate(pts):
        if i<len(conf) and conf[i]>=threshold:
            color=COLOR_FACE if i<=4 else (0,255,255)
            cv2.circle(frame,tuple(np.round(pt).astype(int)),3,color,-1,cv2.LINE_AA)

def draw_dashboard(frame,people,frame_idx,fps):
    h,w=frame.shape[:2]
    fallen=sum(p["state"]=="FALLEN" for p in people)
    falling=sum(p["state"]=="FALLING" for p in people)
    status="ALERT! FALL DETECTED" if fallen else ("CHECKING FALL" if falling else "NORMAL")
    bg=(0,0,220) if fallen else ((0,145,255) if falling else (0,135,0))
    cv2.rectangle(frame,(0,0),(w,42),(12,18,25),-1)
    cv2.putText(frame,"Fall Detection and Alert",(12,27),cv2.FONT_HERSHEY_SIMPLEX,0.65,(40,255,60),2,cv2.LINE_AA)
    label_w=min(w-10,max(145, int(len(status)*8.5)))
    cv2.rectangle(frame,(w-label_w-5,5),(w-5,37),bg,-1)
    cv2.putText(frame,status,(w-label_w,27),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1,cv2.LINE_AA)
    panel_w=min(190,w-12)
    cv2.rectangle(frame,(6,50),(panel_w,145),(15,22,30),-1)
    cv2.putText(frame,f"PERSONS: {len(people)}",(12,69),cv2.FONT_HERSHEY_SIMPLEX,0.42,(230,230,230),1)
    cv2.putText(frame,f"FALLEN: {fallen}",(12,89),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,255) if fallen else (120,255,120),1)
    cv2.putText(frame,f"FALLING: {falling}",(12,109),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,165,255),1)
    cv2.putText(frame,f"TIME: {frame_idx/max(1.0,fps):.1f}s",(12,129),cv2.FONT_HERSHEY_SIMPLEX,0.42,(220,220,220),1)
