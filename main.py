import argparse
import manim_bg.renderer as rd
import mobjects.mobjects as mbj
import cv2
import os
def main():
    #parsing settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--width",type=int,default= 1920)
    parser.add_argument("--height",type=int,default=1080)
    parser.add_argument("--fps",type=int,default=30)
    parser.add_argument("--input_name",type=str,default="manim_bg/renderer")
    parser.add_argument("--out_name",type=str,default="output.mp4")
    
    args = parser.parse_args()
    renderer = rd.Renderer(args.fps,args.width,args.height,args.out_name)
    f_num = len([f for f in os.listdir(args.input_name) if os.path.isfile(os.path.join(args.input_name, f))])
    for frame_index in range(f_num):
        frame = cv2.imread(os.path.join(args.input_name, f"frame_{frame_index}.png"))
        if frame is not None:
            renderer.video_writer.write(frame)
        else:
            print(f"Warning:Frame {frame_index} not found, skipping.")
    renderer.close_video()

if __name__ == "__main__":
    main()