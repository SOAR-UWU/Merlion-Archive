#include <Arduino.h>
#include <Wire.h>
#include <Servo.h>
#include "MS5837.h"

#define MOTOR_FL 2
#define MOTOR_FR 3
#define MOTOR_BL 6
#define MOTOR_BR 7
#define MOTOR_TA 8

#define PULSE_MIN  1148
#define PULSE_MAX  1832
#define PULSE_ZERO 1488

Servo motor_fl, motor_fr, motor_bl, motor_br, motor_ta;
MS5837 depth_sensor;

void setup() {
  Serial.begin(19200);
  Serial.println("message:Start initialization");

  Wire.begin();
  pinMode(MOTOR_FL, OUTPUT);
  pinMode(MOTOR_FR, OUTPUT);
  pinMode(MOTOR_BL, OUTPUT);
  pinMode(MOTOR_BR, OUTPUT);
  pinMode(MOTOR_TA, OUTPUT);

  motor_fl.attach(MOTOR_FL);
  motor_fr.attach(MOTOR_FR);
  motor_bl.attach(MOTOR_BL);
  motor_br.attach(MOTOR_BR);
  motor_ta.attach(MOTOR_TA);

  Serial.println("message:Start arming motors");

  motor_fl.writeMicroseconds(PULSE_MAX);
  motor_fr.writeMicroseconds(PULSE_MAX);
  motor_bl.writeMicroseconds(PULSE_MAX);
  motor_br.writeMicroseconds(PULSE_MAX);
  motor_ta.writeMicroseconds(PULSE_MAX);
  delay(500);

  motor_fl.writeMicroseconds(PULSE_ZERO);
  motor_fr.writeMicroseconds(PULSE_ZERO);
  motor_bl.writeMicroseconds(PULSE_ZERO);
  motor_br.writeMicroseconds(PULSE_ZERO);
  motor_ta.writeMicroseconds(PULSE_ZERO);
  delay(500);

  Serial.println("message:Finish arming");

  while (!depth_sensor.init()) {
    Serial.println("message:Init failed!");
    Serial.println("message:Are SDA/SCL connected correctly?");
    Serial.println("message:Blue Robotics Bar30: White=SDA, Green=SCL");
    delay(1000);
  }
  
  Serial.println("message:Sensor initialized");
  depth_sensor.setModel(MS5837::MS5837_30BA);
  depth_sensor.setFluidDensity(997); // kg/m^3 (freshwater, 1029 for seawater)
}

void loop()
{
  // Reads the Serial input (If any) from the tx2, parses it and sends the signals to the motors
  if (Serial.available()) {
    int motor_fl_val = Serial.readStringUntil(',').toInt();
    int motor_fr_val = Serial.readStringUntil(',').toInt();
    int motor_bl_val = Serial.readStringUntil(',').toInt();
    int motor_br_val = Serial.readStringUntil(',').toInt();
    int motor_ta_val = Serial.readStringUntil('\n').toInt();

    motor_fl.writeMicroseconds(motor_fl_val);
    motor_fr.writeMicroseconds(motor_fr_val);
    motor_bl.writeMicroseconds(motor_bl_val);
    motor_br.writeMicroseconds(motor_br_val);
    motor_ta.writeMicroseconds(motor_ta_val);
    
    Serial.flush();
  }

  depth_sensor.read();
  Serial.print("pressure:"); 
  Serial.println(depth_sensor.pressure());
}