"""
Script to create a balanced ISL dataset with 30 basic words and data augmentation.
"""

import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from pathlib import Path

# Configuration
SOURCE_DIR = "/Users/kiran/Downloads/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level"
TARGET_DIR = "/Users/kiran/Desktop/kii/RealTime_ISL_Translator copy/data_Set"
TARGET_IMAGES_PER_CLASS = 50  # Target number of images per class for balance

# 30 Basic/Common ISL Words - selected for everyday communication
BASIC_30_WORDS = [
    "HELLO_HI",       # Greeting
    "THANK",          # Gratitude
    "PLEASE",         # Polite request
    "SORRY",          # Apology
    "WELCOME",        # Welcome
    "GOOD",           # Positive
    "BAD",            # Negative
    "HAPPY",          # Emotion
    "ANGRY",          # Emotion
    "AFRAID",         # Emotion
    "HELP",           # Request
    "STOP",           # Command
    "GO",             # Action
    "COME",           # Action
    "WATER",          # Need
    "FOOD",           # Need
    "HUNGRY",         # Need
    "SLEEP",          # Action
    "UNDERSTAND",     # Communication
    "FRIEND",         # Relationship
    "LIKE",           # Preference
    "WANT",           # Desire
    "NAME",           # Identity
    "HOW",            # Question
    "WHAT",           # Question
    "WHERE",          # Question
    "WHO",            # Question
    "I_ME_MINE_MY",   # Pronoun
    "YOU",            # Pronoun
    "THINK"           # Mental action
]


def create_target_directory():
    """Create the target data_Set directory structure."""
    if os.path.exists(TARGET_DIR):
        print(f"Removing existing directory: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Created directory: {TARGET_DIR}")


def get_source_images(word):
    """Get all image files from a word folder."""
    word_path = os.path.join(SOURCE_DIR, word)
    
    if not os.path.exists(word_path):
        print(f"Warning: Word folder not found: {word}")
        return []
    
    images = []
    for f in os.listdir(word_path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(os.path.join(word_path, f))
    
    return images


def augment_image(image_path, output_path, aug_type):
    """Apply augmentation to an image and save it."""
    try:
        img = Image.open(image_path)
        
        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if aug_type == 'flip':
            # Horizontal flip
            img = ImageOps.mirror(img)
        
        elif aug_type == 'brightness_up':
            # Increase brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(1.1, 1.3))
        
        elif aug_type == 'brightness_down':
            # Decrease brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.7, 0.9))
        
        elif aug_type == 'contrast':
            # Adjust contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        elif aug_type == 'rotate_small':
            # Small rotation (-15 to 15 degrees)
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=(0, 0, 0), expand=False)
        
        elif aug_type == 'color':
            # Color adjustment
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        elif aug_type == 'sharpness':
            # Sharpness adjustment
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.5))
        
        elif aug_type == 'zoom':
            # Center crop and resize (zoom effect)
            width, height = img.size
            crop_factor = random.uniform(0.85, 0.95)
            new_width = int(width * crop_factor)
            new_height = int(height * crop_factor)
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            img = img.crop((left, top, right, bottom))
            img = img.resize((width, height), Image.LANCZOS)
        
        elif aug_type == 'noise':
            # Add slight gaussian noise
            img_array = np.array(img)
            noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        elif aug_type == 'combined1':
            # Combination: flip + brightness
            img = ImageOps.mirror(img)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.9, 1.1))
        
        elif aug_type == 'combined2':
            # Combination: rotation + contrast
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, fillcolor=(0, 0, 0), expand=False)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.9, 1.1))
        
        img.save(output_path, 'JPEG', quality=95)
        return True
    
    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")
        return False


def process_word(word, word_index):
    """Process a single word: copy original images and augment to reach target count."""
    print(f"\n[{word_index+1}/30] Processing: {word}")
    
    # Get source images
    source_images = get_source_images(word)
    
    if not source_images:
        print(f"  Skipping {word} - no images found")
        return 0
    
    # Create word directory in target
    word_dir = os.path.join(TARGET_DIR, word)
    os.makedirs(word_dir, exist_ok=True)
    
    original_count = len(source_images)
    print(f"  Found {original_count} original images")
    
    # If we have more than target, randomly sample to balance
    if original_count > TARGET_IMAGES_PER_CLASS:
        print(f"  Downsampling from {original_count} to {TARGET_IMAGES_PER_CLASS}")
        source_images = random.sample(source_images, TARGET_IMAGES_PER_CLASS)
    
    # Copy original images (or sampled subset)
    image_count = 0
    for idx, src_img in enumerate(source_images):
        dst_path = os.path.join(word_dir, f"{word}_{idx+1:03d}.jpg")
        try:
            # Open and save to ensure consistent format
            img = Image.open(src_img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(dst_path, 'JPEG', quality=95)
            image_count += 1
        except Exception as e:
            print(f"  Error copying {src_img}: {e}")
    
    print(f"  Copied {image_count} images")
    
    # Calculate how many augmented images we need
    augment_needed = TARGET_IMAGES_PER_CLASS - image_count
    
    if augment_needed <= 0:
        print(f"  Dataset balanced at {image_count} images")
        return image_count
    
    print(f"  Need to generate {augment_needed} augmented images")
    
    # Augmentation types to cycle through
    aug_types = [
        'flip', 'brightness_up', 'brightness_down', 'contrast',
        'rotate_small', 'color', 'sharpness', 'zoom', 'noise',
        'combined1', 'combined2'
    ]
    
    aug_count = 0
    aug_type_idx = 0
    
    while aug_count < augment_needed:
        # Cycle through source images and augmentation types
        for src_img in source_images:
            if aug_count >= augment_needed:
                break
            
            aug_type = aug_types[aug_type_idx % len(aug_types)]
            output_path = os.path.join(word_dir, f"{word}_aug_{image_count + aug_count + 1:03d}.jpg")
            
            if augment_image(src_img, output_path, aug_type):
                aug_count += 1
            
            aug_type_idx += 1
    
    total_images = image_count + aug_count
    print(f"  Generated {aug_count} augmented images")
    print(f"  Total images for {word}: {total_images}")
    
    return total_images


def create_dataset_summary(class_counts):
    """Create a summary file for the dataset."""
    summary_path = os.path.join(TARGET_DIR, "dataset_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ISL Dataset Summary - 30 Basic Words\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Target images per class: {TARGET_IMAGES_PER_CLASS}\n")
        f.write(f"Total classes: {len(class_counts)}\n")
        f.write(f"Total images: {sum(class_counts.values())}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("Class-wise distribution:\n")
        f.write("-" * 40 + "\n")
        
        for word, count in sorted(class_counts.items()):
            f.write(f"  {word:20s}: {count:4d} images\n")
        
        f.write("\n" + "=" * 60 + "\n")
        
        # Statistics
        counts = list(class_counts.values())
        f.write(f"\nStatistics:\n")
        f.write(f"  Min images per class: {min(counts)}\n")
        f.write(f"  Max images per class: {max(counts)}\n")
        f.write(f"  Average per class: {sum(counts)/len(counts):.1f}\n")
    
    print(f"\nDataset summary saved to: {summary_path}")


def main():
    """Main function to create the balanced dataset."""
    print("=" * 60)
    print("Creating Balanced ISL Dataset with 30 Basic Words")
    print("=" * 60)
    print(f"\nSource directory: {SOURCE_DIR}")
    print(f"Target directory: {TARGET_DIR}")
    print(f"Target images per class: {TARGET_IMAGES_PER_CLASS}")
    print(f"\nSelected 30 basic words:")
    for i, word in enumerate(BASIC_30_WORDS, 1):
        print(f"  {i:2d}. {word}")
    
    # Create target directory
    create_target_directory()
    
    # Process each word
    class_counts = {}
    for idx, word in enumerate(BASIC_30_WORDS):
        count = process_word(word, idx)
        if count > 0:
            class_counts[word] = count
    
    # Create summary
    print("\n" + "=" * 60)
    print("Dataset Creation Complete!")
    print("=" * 60)
    
    create_dataset_summary(class_counts)
    
    print(f"\nTotal classes created: {len(class_counts)}")
    print(f"Total images: {sum(class_counts.values())}")
    print(f"\nDataset saved to: {TARGET_DIR}")


if __name__ == "__main__":
    main()

