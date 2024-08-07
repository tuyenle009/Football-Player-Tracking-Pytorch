import torch
from torchvision import transforms
import argparse
import cv2
from src.classifier import PlayerClassifier
from src.tracker import Tracker
from tqdm import tqdm

def get_args():
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Detector-Classifier pipeline")
    # Add arguments for detector and classifier checkpoints, video path, and output path
    parser.add_argument("--detector_checkpoint", "-d", type=str, default="checkpoints/player_detector.pt",
                        help="Path to the detector model checkpoint")
    parser.add_argument("--classifier_checkpoint", "-c", type=str, default="checkpoints/player_classification.pt",
                        help="Path to the classifier model checkpoint")
    parser.add_argument("--video_path", "-v", type=str, default="videos/football.mp4",
                        help="Path to the input video file")
    parser.add_argument("--output_path", "-o", type=str, default="demo/output.mp4",
                        help="Path to save the output video file")
    # Parse and return the arguments
    args = parser.parse_args()
    return args



def inference(args):
    # Select the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Open the video file
    cap = cv2.VideoCapture(args.video_path)

    # Load the YOLOv5 detector model
    detector = torch.hub.load('yolov5', 'custom', args.detector_checkpoint, source='local')
    # Initialize the player classifier
    classifier = PlayerClassifier()
    # Load the saved classifier checkpoint
    checkpoint = torch.load(args.classifier_checkpoint)
    classifier.load_state_dict(checkpoint["state_dict"])
    # Set models to evaluation mode
    detector.to(device).eval()
    classifier.to(device).eval()

    # Define image transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get video properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Set up the video writer for the output
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
    mask = cv2.imread("images/fb_mask.jpg")
    #tracking players
    tracker = Tracker()
    #set progress bar to see the inference phase
    counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(range(counter), colour='cyan')
    for idx in progress_bar:
        progress_bar.set_description("Frame: {}/{}".format(idx,counter))

        flag, ori_frame = cap.read()
        #create a detection area
        mask_frame = cv2.bitwise_and(ori_frame, mask)
        if not flag:
            break
        # Convert the frame to RGB format
        frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2RGB)

        # Perform detection using the detector
        det_pred = detector(frame, size=1280)
        player_images = []
        # Extract detected player regions and preprocess them
        for coord in det_pred.xyxy[0]:
            xmin, ymin, xmax, ymax, _, _ = coord
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            player_image = frame[ymin:ymax, xmin:xmax, :]
            player_image = transform(player_image)
            player_images.append(player_image)

        # Convert list of player images to a tensor and move to the device
        player_images = torch.stack(player_images).to(device)
        # Perform classification using the classifier
        with torch.no_grad():
            cls_pred = classifier(player_images)
        num_digits, unit_digit, team = cls_pred

        # Get predicted class labels
        num_digits = torch.argmax(num_digits, dim=1)
        unit_digit = torch.argmax(unit_digit, dim=1)
        team = torch.argmax(team, dim=1)
        # Draw bounding boxes and annotations on the original frame
        for (xmin, ymin, xmax, ymax, _, _), n, u, t in zip(det_pred.xyxy[0], num_digits, unit_digit, team):
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            u = u.item()
            if n == 2 or u == 10:
                jersey_number = 0
            else:
                if n == 0:  # 1 digit number
                    if u == 0:
                        jersey_number = 0
                    else:
                        jersey_number = u
                else:  # 2 digit number
                    jersey_number = u + 10
            if t == 0:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)
            # Draw ellipse for the player
            ori_frame= tracker.draw_annotations(ori_frame, bbox, color, jersey_number)
        # Write the annotated frame to the output video
        out.write(ori_frame)
    # Release video capture and writer resources
    cap.release()
    out.release()

if __name__ == '__main__':
    # Get command-line arguments and run the inference function
    args = get_args()
    inference(args)