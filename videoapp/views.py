import os
from django.shortcuts import render, redirect
from django.http import FileResponse
from .forms import VideoUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
import cv2
import cvlib as cv
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import uuid

# Load gender detection model globally
model = load_model('gender_detection.h5')

def render_home(request):
    return render(request, 'videoapp/index.html')

def process_video(input_video_path, output_video_path):
    # Open video file
    video = cv2.VideoCapture(input_video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter object for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    classes = ['man', 'woman']
    # Load pre-trained MobileNet SSD for person detection
    person_net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

    # Initialize variables
    person_count = 0
    max_count = 0
    count_per = []  # List to store person count for each frame

    while video.isOpened():
        status, frame = video.read()

        if not status:
            break

        # Blob for person detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        person_net.setInput(blob)
        detections = person_net.forward()

        # Reset person count for the current frame
        person_count = 0
        women_count=0


        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])

            if confidence > 0.5 and idx == 15:  # Class ID 15 is 'person'
                # Bounding box for person detection
                box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw person bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                person_count += 1

                # Label person count on the bounding box
                label = f'Person {person_count}'
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Face detection within the person region
                person_region = frame[startY:endY, startX:endX]
                if person_region.size == 0:
                    continue  # Skip if region is empty

                face, _ = cv.detect_face(person_region)
                for f in face:
                    (fX, fY, fW, fH) = f
                    face_rect = (startX + fX, startY + fY, startX + fW, startY + fH)
                    (f_startX, f_startY, f_endX, f_endY) = face_rect

                    # Ensure valid face crop dimensions before resizing
                    if f_startX < 0 or f_startY < 0 or f_endX > frame_width or f_endY > frame_height:
                        continue

                    face_crop = np.copy(frame[f_startY:f_endY, f_startX:f_endX])

                    # Check if the face region is valid and non-empty
                    if face_crop.size == 0:
                        continue

                    face_crop = cv2.resize(face_crop, (96, 96))  # Resize face to 96x96 for model input
                    face_crop = face_crop.astype("float") / 255.0
                    face_crop = img_to_array(face_crop)
                    face_crop = np.expand_dims(face_crop, axis=0)

                    # Predict gender
                    conf = model.predict(face_crop)[0]
                    gender_label = classes[np.argmax(conf)]
                    gender_text = f"{gender_label}"
                    if gender_label == "women":
                        women_count += 1

                    # Put gender label on the frame
                    cv2.putText(frame, gender_text, (f_startX + 20, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Store the person count for the current frame in the list
        count_per.append(person_count)

        # Show total person count on the top-right corner of the frame
        total_count_label = f'Total Persons: {person_count}'
        cv2.putText(frame, total_count_label, (frame_width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show total person count on the top-right corner of the frame
        t=True if women_count==1 else False
        #lone_woman = f'Lone Woman: {t}'
        # cv2.putText(frame, lone_woman, (frame_width - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


        # Write the frame to the output video
        out.write(frame)

        # Keep track of max person count
        if person_count > max_count:
            max_count = person_count

    video.release()
    out.release()
    cv2.destroyAllWindows()

    # Return the list of person counts for later use
    #print(count_per)
    return count_per


def video_upload_view(request):
    if request.method == 'POST':
        # Get the video from the POST request
        video = request.FILES['video']
        input_video_path = os.path.join(settings.INPUT_VIDEO_PATH, video.name)
        output_video_name = str(uuid.uuid4()) + '.mp4'
        output_video_path = os.path.join(settings.OUTPUT_VIDEO_PATH, output_video_name)

        # Save uploaded video
        with open(input_video_path, 'wb+') as destination:
            for chunk in video.chunks():
                destination.write(chunk)

        # Process video (using your process_video function)
        process_video(input_video_path, output_video_path)

        # Return the processed video for download
        return FileResponse(open(output_video_path, 'rb'), as_attachment=True, filename=output_video_name)

    return render(request, 'videoapp/upload.html')

import folium
from django.shortcuts import render

# Define a view to render the Folium map
def show_map(request):
    # Create a base map centered around India
    india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    # List of existing locations with their coordinates (danger areas)
    locations = {
        "Nungambakkam High Road": (13.0600, 80.2394),
        "Kodambakkam High Road": (13.0532, 80.2355),
        "Kotturpuram MRTS Station": (13.0105, 80.2391),
        "OMR (Old Mahabalipuram Road)": (12.8700, 80.2209),
        "Durgabai Deshmukh Road in Adyar": (13.0066, 80.2560),
        "Access road to Kodambakkam station": (13.0517, 80.2294),
        "Road next to Central station": (13.0826, 80.2780),
        "Subway from Tirusulam station to airport": (12.9820, 80.1638),
        "Swami Sivananda Salai (Mount Road to Beach Road)": (13.0817, 80.2830),
        "Mambalam": (13.0330, 80.2224),
        "Royapettah": (13.0541, 80.2664),
        "Taramani": (12.9765, 80.2391),
    }

    # List of new cities with their coordinates (danger areas)
    cities = {
        "Delhi": (28.6139, 77.2090),
        "Surat": (21.1702, 72.8311),
        "Ahmedabad": (23.0225, 72.5714),
        "Jaipur": (26.9124, 75.7873),
        "Kochi": (9.9312, 76.2673),
        "Indore": (22.7196, 75.8577),
        "Patna": (25.5941, 85.1376),
        "Nagpur": (21.1458, 79.0882),
        "Coimbatore": (11.0168, 76.9558),
        "Kozhikode": (11.2588, 75.7804),
    }

    # Add markers to the map for existing locations (danger areas)
    for location, coordinates in locations.items():
        folium.Marker(
            location=coordinates,
            popup=f"Danger Area: {location}",
            icon=folium.Icon(color="darkred", icon="exclamation-sign"),
        ).add_to(india_map)

    # Add markers to the map for new cities with different icon
    for city, coordinates in cities.items():
        folium.Marker(
            location=coordinates,
            popup=f"Danger Area: {city}",
            icon=folium.Icon(color="orange", icon="exclamation-sign"),
        ).add_to(india_map)

    # Convert map to HTML representation
    map_html = india_map._repr_html_()

    # Render the map in the template
    return render(request, 'videoapp/map_page.html', {'map_html': map_html})


import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from django.http import FileResponse
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

def gesture_view(request):
    if request.method == 'POST':
        # Handle file upload
        uploaded_file = request.FILES['video_file']
        input_path = os.path.join(settings.MEDIA_ROOT, 'input_gesture', uploaded_file.name)
        output_path = os.path.join(settings.MEDIA_ROOT, 'output_gesture', 'output_video.mp4')
        
        # Save uploaded video to input_gesture folder
        default_storage.save(input_path, uploaded_file)

        # Run the gesture detection script
        run_gesture_detection(input_path, output_path)
        
        # Return the output video for download
        return FileResponse(open(output_path, 'rb'), as_attachment=True, filename='output_video.mp4')

    return render(request, 'videoapp/gesture.html')

def run_gesture_detection(input_path, output_path):
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions

    df = pd.read_csv('coordinates_final.csv')
    X = df.drop('class', axis=1)  # Features
    y = df['class']  # Target

    with open('gesture.pkl', 'rb') as f:
        model = pickle.load(f)

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Start the holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if no more frames

            # Convert the BGR frame to RGB for Mediapipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the frame for landmark detection
            results = holistic.process(image)

            # Convert back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks on the frame
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

            # 2. Right hand landmarks
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

            # 3. Left hand landmarks
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

            # 4. Pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Perform gesture classification (try-catch to handle empty frames or lack of landmarks)
            try:
                # Extract pose and face landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concatenate pose and face data
                row = pose_row + face_row

                # Make predictions
                X = pd.DataFrame([row])
                gesture_class = model.predict(X)[0]
                gesture_prob = model.predict_proba(X)[0]

                # Display the gesture class and probability
                coordinates = tuple(np.multiply(
                    np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                              results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                    [640, 480]).astype(int))

                cv2.rectangle(image, (coordinates[0], coordinates[1]+5), 
                              (coordinates[0]+len(gesture_class)*20, coordinates[1]-30), 
                              (245, 117, 16), -1)
                cv2.putText(image, gesture_class, coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(round(gesture_prob[np.argmax(gesture_prob)], 2)), 
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error during classification: {e}")
                pass

            # Write the frame to the output video
            out.write(image)

            # Display the frame (remove for headless environments)
            cv2.imshow('Raw Webcam Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

from django.shortcuts import render
import pandas as pd
import plotly.express as px

def dashboard_view(request):
    # Load your data
    data = pd.read_csv('crimes_against_women_2001-2014.csv')

    # Create a Plotly bar chart
    fig = px.bar(
        data_frame=data,
        x="STATE/UT",
        y=[
            'Rape', 
            'Kidnapping and Abduction', 
            'Assault on women with intent to outrage her modesty', 
            'Insult to modesty of Women', 
            'Importation of Girls'
        ],
        title="Crime Against Women by State/UT",
        color_discrete_sequence=['#1f77b4']  # Specify the color here
    )

    # Serialize the figure to HTML
    plot_html = fig.to_html(full_html=False)

    return render(request, 'videoapp/dashboard.html', {'plot_html': plot_html})

def gesture_redirect(request):
    return redirect

from django.shortcuts import render

def contact_us_view(request):
    return render(request, 'videoapp/contact.html')