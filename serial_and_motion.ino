#define ENA 9
#define ENB 10
#define IN1 7
#define IN2 8
#define IN3 5
#define IN4 6

float received = 5.0;  // Initialize with a value to stop motors initially

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  
  analogWrite(ENA, 100);  // Set motor speed (PWM)
  analogWrite(ENB, 100);  // Set motor speed (PWM)
  
  Serial.begin(9600); // Initialize serial communication

  // Print a message that setup is complete (for debugging)
  Serial.println("Setup complete");
}

void loop() {
  // Check if there is data available to read from Serial
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); // Read incoming string
    received = command.toFloat(); // Convert string to float
    delay(50);  // Small delay to allow full reading
  }

  // Debug: Print received value to Serial Monitor
  Serial.print("Received: ");
  Serial.println(received);

  // Control logic based on received value
  if (received >= 0.0 && received <= 0.45) {
    // Move motors forward to the right
    Serial.println("Moving right...");
    digitalWrite(IN3, HIGH); // left motor moved 
    digitalWrite(IN4, LOW);
    digitalWrite(IN1, LOW);  // right motor stopped
    digitalWrite(IN2, LOW);
    analogWrite(ENB, 255);   // Full speed for left motor
    analogWrite(ENA, 0);     // stopp right motor 
  } 
  else if (received >= 0.55 && received <= 1.0) {
    // Move motors forward to the left
    Serial.println("Moving left...");
    digitalWrite(IN1, HIGH); // right motor moved
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW);  // right motor moved
    digitalWrite(IN4, LOW);
    analogWrite(ENA, 255);   // move right motor
    analogWrite(ENB, 0);     // stop left motor 
  } 
  else if (received >0.45 && received <0.55 ) {
    // Move motors forward 
    Serial.println("Moving forward...");
    digitalWrite(IN1, HIGH); // Right motor forward
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH);  // Left motor forward
    digitalWrite(IN4, LOW);
    analogWrite(ENA, 255);   // Full speed for right motor
    analogWrite(ENB, 255);     // Full speed for left motor
  }
  else {
    // Stop motors if the value is outside the valid range
    Serial.println("Stopping motors...");
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
  }

  delay(100); // Small delay to avoid flooding serial communication
}

