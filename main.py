#!/usr/bin/env python3
"""AI CCTV fall detection pipeline: YOLO Pose + temporal state tracking."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from fall_detection import (
    FallKinematicAnalyzer, PersonFallTracker, draw_skeleton,
    draw_dashboard, FallEventLogger
)

def parse_source(src):
    return int(src) if src.isdigit() else src

def choose_device(config_value="auto"):
    if config_value != "auto":
        return config_value
    return "0" if torch.cuda.is_available() else "cpu"

def main():
    ap=argparse.ArgumentParser(description="CCTV Fall Detection and Alert")
    ap.add_argument("--source",required=True,help="Video file, RTSP URL, or camera index (0)")
    ap.add_argument("--config",default="config/settings.json")
    ap.add_argument("--weights",default=None,help="YOLO pose weights path")
    ap.add_argument("--output",default="output/fall_detection.mp4")
    ap.add_argument("--no-display",action="store_true")
    args=ap.parse_args()

    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    weights=args.weights or cfg.get("weights","yolo11n-pose.pt")
    device=choose_device(cfg.get("device","auto"))
    model=YOLO(weights)
    cap=cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {args.source}")

    fps=cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps<1 or fps>120: fps=float(cfg.get("fps_fallback",25))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path=Path(args.output); out_path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(out_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    analyzer=FallKinematicAnalyzer(keypoint_confidence=float(cfg.get("keypoint_confidence",0.30)))
    trackers={}
    logger=FallEventLogger(Path(cfg.get("event_log","output/fall_events.jsonl")))
    frame_idx=0
    print(f"[INFO] Loading {weights} on {device}; source={args.source}; fps={fps:.1f}")
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frame_idx+=1
            result=model.track(
                frame,persist=True,tracker="bytetrack.yaml",conf=float(cfg.get("confidence",0.35)),
                classes=[0],device=device,verbose=False
            )[0]
            people=[]
            boxes=result.boxes
            kps=result.keypoints
            if boxes is not None and kps is not None:
                xy_all=kps.xy.cpu().numpy()
                conf_all=kps.conf.cpu().numpy() if kps.conf is not None else np.ones(xy_all.shape[:2],dtype=np.float32)
                for i,box in enumerate(boxes):
                    x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy())
                    tid=int(box.id[0].cpu()) if box.id is not None else i
                    pose=analyzer.analyze_pose(xy_all[i],conf_all[i],(x1,y1,x2,y2))
                    state=trackers.setdefault(tid,PersonFallTracker(tid,fps=fps)).update(
                        (x1,y1,x2,y2),pose
                    )
                    people.append({"track_id":tid,"box":(x1,y1,x2,y2),"state":state,
                                   "pose":pose,"keypoints":xy_all[i],"keypoint_conf":conf_all[i]})
                    if state=="FALLEN":
                        logger.log(tid,frame_idx,fps,pose)
            # Expire trackers that have not been seen for a while.
            active={p["track_id"] for p in people}
            for tid in list(trackers):
                if tid not in active:
                    trackers[tid].mark_missed()
                    if trackers[tid].missed_frames>int(fps*5):
                        del trackers[tid]

            annotated=frame.copy()
            for person in people:
                draw_skeleton(annotated,person["keypoints"],person["keypoint_conf"],
                              float(cfg.get("keypoint_confidence",0.30)))
                x1,y1,x2,y2=person["box"]
                state=person["state"]
                color={"STANDING":(0,220,90),"FALLING":(0,165,255),
                       "FALLEN":(0,0,255),"RECOVERING":(255,190,0)}.get(state,(200,200,200))
                cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                cv2.putText(annotated,f"#{person['track_id']} {state}",(x1,max(20,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2,cv2.LINE_AA)
            fallen=sum(1 for p in people if p["state"]=="FALLEN")
            falling=sum(1 for p in people if p["state"]=="FALLING")
            draw_dashboard(annotated,people,frame_idx,fps)
            writer.write(annotated)
            if not args.no_display:
                cv2.imshow("Fall Detection and Alert",annotated)
                if cv2.waitKey(1)&0xFF==ord("q"): break
    finally:
        cap.release(); writer.release(); cv2.destroyAllWindows()
    logger.close()
    print(f"[DONE] Annotated output: {out_path}")
    print(f"[DONE] Fall events: {logger.event_count}")

if __name__=="__main__":
    main()
