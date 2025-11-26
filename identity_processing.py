import cv2
import dlib
import numpy as np
import os

# ✅ Load heavy models ONCE (on module import)
face_detector = dlib.get_frontal_face_detector()

# ✅ WINDOWS COMPATIBLE: Use relative paths
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "face_reco", "models")

# Load face recognition model
face_encoder_path = os.path.join(models_dir, "dlib_face_recognition_resnet_model_v1.dat")
if not os.path.exists(face_encoder_path):
    print(f"⚠️  WARNING: Face encoder model not found at {face_encoder_path}")
    print("   Face recognition will not work!")
    face_encoder = None
else:
    face_encoder = dlib.face_recognition_model_v1(face_encoder_path)
    print(f"✅ Loaded face encoder from {face_encoder_path}")

# Load shape predictor
shape_predictor_path = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(shape_predictor_path):
    print(f"⚠️  WARNING: Shape predictor not found at {shape_predictor_path}")
    print("   Face recognition will not work!")
    shape_predictor = None
else:
    shape_predictor = dlib.shape_predictor(shape_predictor_path)
    print(f"✅ Loaded shape predictor from {shape_predictor_path}")

# ✅ WINDOWS COMPATIBLE: Base directory for known faces
base_dir = os.path.join(script_dir, "face_reco", "images")
print(f"📂 Face images directory: {base_dir}")


def load_people_images(base_dir):
    """Return dict of person_name -> list of image paths."""
    if not os.path.exists(base_dir):
        print(f"⚠️  WARNING: Images directory not found: {base_dir}")
        return {}
    
    people = {}
    for person_name in os.listdir(base_dir):
        folder = os.path.join(base_dir, person_name)
        if os.path.isdir(folder):
            image_paths = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ]
            if image_paths:
                people[person_name] = image_paths
                print(f"   Found {len(image_paths)} image(s) for {person_name}")
    
    if not people:
        print(f"⚠️  WARNING: No face images found in {base_dir}")
        print("   Create folders with photos: face_reco/images/Your_Name/photo1.jpg")
    
    return people


def get_face_descriptor(image_paths):
    """Compute average descriptor for a list of face images."""
    if face_encoder is None or shape_predictor is None:
        return None
    
    descriptors = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"   ⚠️  Could not read image: {path}")
            continue
        faces = face_detector(img, 1)
        if faces:
            shape = shape_predictor(img, faces[0])
            descriptor = np.array(face_encoder.compute_face_descriptor(img, shape))
            descriptors.append(descriptor)
        else:
            print(f"   ⚠️  No face detected in: {path}")
    
    return np.mean(descriptors, axis=0) if descriptors else None


def preload_known_faces():
    """Preload known faces ONCE."""
    print("\n" + "=" * 60)
    print("👤 Loading known faces...")
    print("=" * 60)
    
    if face_encoder is None or shape_predictor is None:
        print("❌ Face recognition models not loaded. Cannot preload faces.")
        return {}
    
    people = load_people_images(base_dir)
    known_faces = {}
    
    for name, paths in people.items():
        print(f"Processing {name}...")
        desc = get_face_descriptor(paths)
        if desc is not None:
            known_faces[name] = desc
            print(f"  ✅ Loaded {name}")
        else:
            print(f"  ❌ Failed to load {name}")
    
    print("=" * 60)
    print(f"✅ Total known faces loaded: {len(known_faces)}")
    print("=" * 60)
    print()
    
    if len(known_faces) == 0:
        print("⚠️  WARNING: No faces were loaded!")
        print("   Make sure you have:")
        print(f"   1. Photos in: {base_dir}\\YourName\\")
        print(f"   2. Model files in: {models_dir}")
        print()
    
    return known_faces


# ✅ Keep global cache
known_faces = preload_known_faces()


def process_identity_from_frame(frame):
    """Recognize identity from frame using preloaded models and faces."""
    if face_encoder is None or shape_predictor is None:
        return "Unknown"
    
    if len(known_faces) == 0:
        return "Unknown"
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    identified_person = "Unknown"

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        shape = shape_predictor(frame, dlib.rectangle(x, y, x + w, y + h))
        descriptor = np.array(face_encoder.compute_face_descriptor(frame, shape))

        identity, min_distance = "Unknown", 0.6  # threshold (lower = stricter)

        for name, stored_desc in known_faces.items():
            distance = np.linalg.norm(stored_desc - descriptor)
            if distance < min_distance:
                min_distance = distance
                identity = name

        identified_person = identity

    return identified_person


# Test on import
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING FACE RECOGNITION")
    print("=" * 60)
    print()
    
    if len(known_faces) > 0:
        print("✅ Face recognition is ready!")
        print(f"   Known people: {list(known_faces.keys())}")
    else:
        print("❌ Face recognition is NOT ready!")
        print()
        print("To fix:")
        print(f"1. Add photos to: {base_dir}\\YourName\\")
        print("2. Make sure model files exist in:", models_dir)
        print("3. Restart the face recognition service")
    print()
