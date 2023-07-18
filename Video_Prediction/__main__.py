import argparse

from RealtimeTest import start

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This Programm takes a video- and a model-path and does a semantic segmentation in realtime."
    )

    parser.add_argument("video_path", help=" Path to the video", default="")
    parser.add_argument("model_path", help=" Path to the model for prediction")

    args = parser.parse_args()
    print("Load video from " + args.video_path + " and segment it with model " + args.model_path)

    start(video_path=args.video_path, model_path=args.model_path)
