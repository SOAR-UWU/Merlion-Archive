/*
 * Written by : Clarence (ROS Integration) , Lim Pin (Hardware Interfacing)
 * Last Editted : 10/21/21
 *
 * The code below is used for SAUVC Motor control system , this board will be used to communicate 
 * with higher command ,Control motor speed as well as power ,and perform pressure readings for ros,
 * emergency power cut off will only be executed by the physical switch
 *
 *Hardware layout of the board for reference  : https://easyeda.com/editor#id=287c755b24c842f48219941e13294614
*/

#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <Servo.h>
#include <Wire.h>
#include "MS5837.h"
#include <sensor_msgs/FluidPressure.h>
#include <merlion_hardware/Motor.h>


MS5837 sensor;

#define MOTORSIZE 5 // Change this if this physical motor count increases
#define M1_PIN 23
#define M2_PIN 3
#define M3_PIN 17
#define M4_PIN 22
#define M5_PIN 4


struct MotorSpec 
{
   int pinNo;
   int armingFreq; 
   Servo esc; 
};


//state variables 
bool robotActive = false;
bool LED_BUILTINSTATE;

///=============Function declaration=======================================
void motorCommandCallback(const merlion_hardware::Motor& motorVal );

//ROS declarations 
ros::NodeHandle node_handle; //rosserial object 

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

std_msgs::Float32 sensorDepth; 
ros::Publisher depthPub("merlion_hardware/sensor_depth", &sensorDepth);

void setRobotState(const std_msgs::UInt16 &motorVal);
ros::Subscriber<std_msgs::UInt16> robotStateSubscriber("/merlion_init/robot_state", &setRobotState);

std_msgs::Bool isMotorActive; 
ros::Publisher motorActiveStatePub("/merlion_init/motor_active", &isMotorActive);

void setup() 
{
  Serial.begin(9600);
  rosInit();
  armAll();
  //TODO: find a way to cut off comms when mother controller is not connected
  //pressureSensorInit();
  toggleLEDBuiltIn();
}

int i = 100;
bool flag = true;

void loop() 
{
  if (!robotActive)
  {
    i = 10;
    int speedPercentage[] = {i,i,i,i,i}; 
    motorSetSpeedAllPercentage(speedPercentage);
    delay(100);
    node_handle.spinOnce();
    toggleLEDBuiltIn();
    return;
  }

  sensor.read();
  pressureData.fluid_pressure = sensor.pressure() ;  
  Serial.println(sensor.pressure());
  pressurePub.publish(&pressureData);
  sensorDepth.data = sensor.depth(); 
  depthPub.publish( &sensorDepth );
  node_handle.spinOnce();
  

//  if (i == 100){
//    flag = false; 
//  }else if(i == -100){
//    flag = true;
//  }
//
//  if (flag){
//    i += 1;
//  }else{
//    i -= 1;
//  }
//  
//  int speedPercentage[] = {i,i,i,i,i}; 
//  motorSetSpeedAllPercentage(speedPercentage);


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
void armAll()
{
  Serial.println("Arming Motors");
  for(int i = 0 ; i<MOTORSIZE ; i++)
  {
    motors[i].pinNo = motorPins[i];
    motors[i].armingFreq = armingFreq[i];
    motors[i].esc.attach(motorPins[i]);

    motors[i].esc.writeMicroseconds(motors[i].armingFreq);
    ///BLOCKING CODE 
    delay(8000);
    ///BLOCKING CODE 
    Serial.print(motors[i].pinNo);Serial.print(" ");Serial.println(motors[i].armingFreq);
  }
  Serial.println("MOTORS ARMED !!!");
  isMotorActive.data = true; 
  for (int i=0; i < 10; i++)
  {
    motorActiveStatePub.publish(&isMotorActive);
    node_handle.spinOnce();
    delay(100);
  }
  
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
  node_handle.initNode();
  node_handle.advertise(pressurePub);
  node_handle.advertise(depthPub);
  node_handle.advertise(motorActiveStatePub);
  node_handle.subscribe(motor_Subscriber);
  node_handle.subscribe(robotStateSubscriber);
  
  //Initialise Led blinking sequance
  pinMode(LED_BUILTIN,OUTPUT);
  digitalWrite(LED_BUILTIN ,HIGH);
  LED_BUILTINSTATE = true;
  delay(2000);
}
///============================ROS CALLBACK CODE ===================================
//TODO : Add ROSSERIAL CODE HERE
void motorCommandCallback(const merlion_hardware::Motor& motorVal )
{
  toggleLEDBuiltIn();
  int m1 = motorVal.m1; 
  int m2 = motorVal.m2;
  int m3 = motorVal.m3;
  int m4 = motorVal.m4;
  int m5 = motorVal.m5;

  /// Callback function for motor / pressure sensore return 
  motorSetSpeedFreq(motors[0],freqConversion(m1));
  motorSetSpeedFreq(motors[1],freqConversion(m2));
  motorSetSpeedFreq(motors[2],freqConversion(m3));
  motorSetSpeedFreq(motors[3],freqConversion(m4));
  motorSetSpeedFreq(motors[4], freqConversion(m5));
  toggleLEDBuiltIn();
}

void setRobotState(const std_msgs::UInt16& robotState)
{
    if (robotState.data == 1)
    {
      robotActive = true;
    }else{
      robotActive = false;
    }
}  

///============================MOTOR CONTROL CODE ==================================
void motorSetSpeedAllPercentage(int _speedPercentage [])
{
  //TAKE NOTE : MAKE SURE THAT _speedPercentage ARRAY SIZE IS EQUALS TO MOTORSIZE
  for(int i =0 ;i<MOTORSIZE ; i++)
  {
    motors[i].esc.writeMicroseconds(freqConversion(_speedPercentage[i]));  
 
    //Serial.print("Motor Speed change : Motor : "); Serial.print(i); Serial.print(" speedPercentage : ");Serial.println(_speedPercentage[i]);
  }
}
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
int freqConversion(int speedPercentage)
{
 if(speedPercentage >= 0)
 {
   //CW code : 
   int freq = map(speedPercentage,0,100,generalArmingFreq,1900);
   return freq;
 }
 else if(speedPercentage <=0 )
 {
   //ACW code : 
   int freq = map(-speedPercentage,0,100,generalArmingFreq,1100);             
   return freq;
 }
 else
 {
   //default code 
   Serial.print("freq :Invalid input : speedPercentage : ");Serial.println(speedPercentage);
 }
}
///============================HELPER CODE ==================================

void toggleLEDBuiltIn()
{
  if(LED_BUILTINSTATE == 1)
  {
    digitalWrite(LED_BUILTIN,LOW );
    LED_BUILTINSTATE = false;
    delay(5); //TODO : REMOVE DELAYS IN INTERNAL FUNCTIONS 
  }
  else 
  {
    digitalWrite(LED_BUILTIN , HIGH);
    LED_BUILTINSTATE = true;
    Serial.println(LED_BUILTINSTATE);
    delay(5);//TODO : REMOVE DELAYS IN INTERNAL FUNCTIONS 
  }
}

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
