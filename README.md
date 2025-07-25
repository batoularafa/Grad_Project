# Grad_Project: Iris-Controlled Smart Wheelchair

**Biomedical Engineering Graduation Project — Alexandria University**

This project provides a smart, hands-free wheelchair system designed for **quadriplegic patients** individuals who have lost motor control in both upper and lower limbs. Using **eye movement and blink detection**, the system enables the user to control wheelchair motion without physical input.

---

## Overview

- **Objective**: Enable paralyzed patients to control a wheelchair using only their eyes.
- **Approach**: Detect eye movements and blinks using OpenCV and MediaPipe, interpret them into directional commands, and send these commands to an Arduino that controls the motors.
- **Safety**: Includes ultrasonic obstacle detection and buzzer alerts.

---

## System Architecture

1. **User's Eye Movement**
  
2. **Raspberry Pi + PiCamera**
   - Captures video feed  
  
3. **MediaPipe (Iris + Blink Detection)**
   - Detects iris position and blinks  
  
4. **Serial Communication (USB)**
   - Sends movement command to Arduino  
  
5. **Arduino**
   - Interprets and executes movement  
  
6. **Motor Driver + Ultrasonic Sensors**
   - Controls wheelchair motion and ensures safety

