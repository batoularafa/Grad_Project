#define ENA 9
#define ENB 10
#define IN1 7
#define IN2 8
#define IN3 5
#define IN4 6
float received;

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  analogWrite(ENA, 100);
  analogWrite(ENB, 100);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    received = command.toFloat();
    delay(50);}
  Serial.println(received);
  delay(50);

  if (received>0.01 && received<=1){
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      // Motor B forward
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
  }
  else{
    digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      // Motor B forward
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
  }
    // switch (command) {
    //   case 'f':
    //     // Move forward
    //     digitalWrite(IN1, HIGH);
    //     digitalWrite(IN2, LOW);
    //     analogWrite(ENA, 255); // Adjust speed
    //     break;
    //   case 'b':
    //     // Move backward
    //     digitalWrite(IN1, LOW);
    //     digitalWrite(IN2, HIGH);
    //     analogWrite(ENA, 255); // Adjust speed
    //     break;
    //   case 's':
    //     // Stop
    //     digitalWrite(IN1, LOW);
    //     digitalWrite(IN2, LOW);
    //     break;
    // }
  }
