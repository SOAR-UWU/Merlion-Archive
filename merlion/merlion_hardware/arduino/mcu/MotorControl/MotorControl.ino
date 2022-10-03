#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt16.h>
#include <Servo.h>
#include <Wire.h>
#include "MS5837.h"
#include <sensor_msgs/FluidPressure.h>
#include <merlion_hardware/Motor.h>


MS5837 sensor;

#define MOTORSIZE 5 // Change this if this physical motor count increases
#define M1_PIN 23
#define M2_PIN 14
#define M3_PIN 16
#define M4_PIN 10
#define M5_PIN 22


struct MotorSpec 
{
   int pinNo;
   int armingFreq; 
   Servo esc; 
};


///=============Function declaration
void motorCommandCallback(const merlion_hardware::Motor& motorVal );

//ROS declarations 
ros::NodeHandle nh; //rosserial object 

std_msgs::String sub_msg; // ros message
std_msgs::UInt16 motorState_msg; // motorState string publisher message

ros::Subscriber<merlion_hardware::Motor> motor_Subscriber("/merlion_hardware/thruster_values", &motorCommandCallback);

MotorSpec motors[MOTORSIZE];
int motorPins [MOTORSIZE] = {M1_PIN,M2_PIN,M3_PIN,M4_PIN,M5_PIN};
int armingFreq[MOTORSIZE] = {1480,1480,1480,1480,1480};
int generalArmingFreq = 1480;
Servo esc;

sensor_msgs::FluidPressure pressureData;
ros::Publisher pressurePub("merlion_hardware/pressure", &pressureData);

void setup() 
{
  Serial.begin(9600);
  armAll();
  //TODO: find a way to cut off comms when mother controller is not connected
  rosInit();
  pressureSensorInit();
}

void loop() 
{
  readAndPubPressureValues();
  delay(100);
}

//============================INIT CODE ==================================

void pressureSensorInit()
{
  Serial.println("Starting");
  
  Wire.begin();

  // Initialize pressure sensor
  // Returns true if initialization was successful
  // We can't continue with the rest of the program unless we can initialize the sensor
  while (!sensor.init()) {
    Serial.println("Init failed!");
    Serial.println("Are SDA/SCL connected correctly?");
    Serial.println("Blue Robotics Bar30: White=SDA, Green=SCL");
    Serial.println("\n\n\n");
    delay(5000);
  }
  Serial.println("sensor done");
  
  sensor.setModel(MS5837::MS5837_30BA);
  sensor.setFluidDensity(997); // kg/m^3 (freshwater, 1029 for seawater)
}

void readAndPubPressureValues()
{
    sensor.read();
    pressureData.fluid_pressure = sensor.pressure();
    pressurePub.publish(&pressureData);
    Serial.println(sensor.depth());
    nh.spinOnce();
}
void armAll()
{
  Serial.println("Arming Motors");
  delay(5000);
  for(int i = 0 ; i<MOTORSIZE ; i++)
  {
    motors[i].pinNo = motorPins[i];
    motors[i].armingFreq = armingFreq[i];
    motors[i].esc.attach(motorPins[i]);

    motors[i].esc.writeMicroseconds(motors[i].armingFreq);
    ///BLOCKING CODE 
    delay(3000);
    ///BLOCKING CODE 
    Serial.print(motors[i].pinNo);Serial.print(" ");Serial.println(motors[i].armingFreq);
  }
  Serial.println("MOTORS ARMED !!!");
}
//

///============================COMMS CODE ==================================
//void serialEvent()
//{
//  //!WARNING : TEMP CODE REPLACE WITH LOOK UP FUCNTION FOR DYNAMIC COMMS METHOD
//  String readValue = Serial.readString(); 
//  //readValue.flush();
//  int value = readValue.toInt(); 
//  Serial.print("ReadValue : ");Serial.println(readValue);
//  Serial.print("ActualValue : ");Serial.println(value);
//}
void rosInit()
{
  nh.initNode();
  nh.advertise(pressurePub);
  nh.subscribe(motor_Subscriber);
  
}
///============================ROS CALLBACK CODE ===================================
//TODO : Add ROSSERIAL CODE HERE
void motorCommandCallback(const merlion_hardware::Motor& motorVal )
{
  int m1 = motorVal.m1; 
  int m2 = motorVal.m2;
  int m3 = motorVal.m3;
  int m4 = motorVal.m4;
  int m5 = motorVal.m5;

  /// Callback function for motor / pressure sensore return 
  motorSetSpeedFreq(motors[0],m1);
  motorSetSpeedFreq(motors[1],m2);
  motorSetSpeedFreq(motors[2],m3);
  motorSetSpeedFreq(motors[3],m4);
  motorSetSpeedFreq(motors[4],m5);
}


///============================MOTOR CONTROL CODE ==================================
//void motorSetSpeedAllPercentage(int _speedPercentage [])
//{
//  //TAKE NOTE : MAKE SURE THAT _speedPercentage ARRAY SIZE IS EQUALS TO MOTORSIZE
//  for(int i =0 ;i<MOTORSIZE ; i++)
//  {
//    motors[i].esc.writeMicroseconds(freqConversion(_speedPercentage[i]));  
//    Serial.print("Motor Speed change : Motor : "); Serial.print(i); Serial.print(" speedPercentage : ");Serial.println(_speedPercentage[i]);
//  }
//}

void motorSetSpeedFreq(MotorSpec _motor,int _speedFreq)
{
  //TODO:SCALE THIS UP TO 4 MOTORS FOR EASE OF CONTROL
  _motor.esc.writeMicroseconds(_speedFreq);
  Serial.print("Motor Speed change : ");Serial.println(_speedFreq);
}
//void motorSetSpeedPercentage(MotorSpec _motor,int _speedPercentage)
//{
//  //TODO:SCALE THIS UP TO 4 MOTORS FOR EASE OF CONTROL 
//  _motor.esc.writeMicroseconds(freqConversion(_speedPercentage));
//  Serial.print("Motor Speed change : ");Serial.println(_speedPercentage);
//}
///============================COMPUTATION CODE ==================================
//int freqConversion(int speedPercentage)
//{
//  if(speedPercentage >= 0)
//  {
//    //CW code : 
//    int freq = map(speedPercentage,0,100,generalArmingFreq,1900 );
//    return freq;
//  }
//  else if(speedPercentage <=0 )
//  {
//    //ACW code : 
//    int freq = map(-speedPercentage,0,100,1100,generalArmingFreq);
//    return freq;
//  }
//  else
//  {
//    //default code 
//    Serial.print("freq :Invalid input : speedPercentage : ");Serial.println(speedPercentage);
//  }
//}

///============================TESTING CODE ==================================
//void testSingle(MotorSpec motor)
//{
//  Serial.println("spinning cw");
//  int delayTime = 1000;
//   for(int i =0 ; i<100 ; i+=10 ) 
//   {
//     Serial.println(i);
//     motorSetSpeedPercentage(motor , i);
//     delay(delayTime);
//   }
//   for(int i =100  ; i>0 ; i-=10 ) 
//   {
//    Serial.println(i);
//    motorSetSpeedPercentage(motor , i);
//    delay(delayTime);
//   }
//}

//void test()
//{
//   Serial.println("spinning cw");
//   int delayTime = 1000;
//   for(int i =0  ; i<100 ; i+=1 ) 
//   {
//     Serial.println(i);
//     int speedPercentage[] = {i,i,i,i,i};
//     motorSetSpeedAllPercentage(speedPercentage);
//
//     delay(delayTime);
//   }
//   for(int i =100  ; i>0 ; i-=1 ) 
//   {
//     Serial.println(i);
//     int speedPercentage[] = {i,i,i,i,i}; 
//     motorSetSpeedAllPercentage(speedPercentage);
//     
//     delay(delayTime);
//   }
//   for(int i =0  ; i>-100 ; i-=1 ) 
//   {
//     Serial.println(i);
//     int speedPercentage[] = {i,i,i,i,i};
//     motorSetSpeedAllPercentage(speedPercentage);
//
//     delay(delayTime);
//   }
//   for(int i =-100  ; i<0 ; i+=1 ) 
//   {
//     Serial.println(i);
//     int speedPercentage[] = {i,i,i,i,i}; 
//     motorSetSpeedAllPercentage(speedPercentage);
//     
//     delay(delayTime);
//   }
//}
