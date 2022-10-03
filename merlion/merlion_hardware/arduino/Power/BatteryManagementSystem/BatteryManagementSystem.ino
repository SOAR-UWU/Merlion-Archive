/*
 * Written by : Lim Pin 
 * Last Editted : 10/21/21
 *
 * The code below is used for SAUVC power control system , this board will be used to communicate 
 * with higher command , read power level of the battery , reading of water leaking and to perform 
 * emergency power cut off if neccesorry
 *
 *Hardware layout of the board for reference  : https://easyeda.com/editor#id=287c755b24c842f48219941e13294614
*/

#define KILL_TOGGLE 3 // For activation of the contactor , acts as an kill switch as well 
//#define KILL_TOGGLE2 4 // For activation of the contactor (acts as a signal booster), acts as an kill switch as well 
//#define KILL_TOGGLE3 5 // For activation of the contactor , acts as an kill switch as well 

String powerStatus = "Initialising";
bool LED_BUILTINSTATE;

void setup() {
  initKillSwitch();
  initComms();

  powerStatus = "Operational";
  updateComms(powerStatus);
}

void loop() {
  // TODO: IMPLEMENT TIMESLICING OPERATION TO ALLOW FOR MULLTI OPERATIONS

  toggleLEDBuiltIn();
  delay(500); //TODO : REMOVE DELAYS IN INTERNAL FUNCTIONS 

}

//=============INIT CODES=======================

void initKillSwitch(void)
{
  pinMode(KILL_TOGGLE,OUTPUT);
  pinMode(LED_BUILTIN , OUTPUT );
  
  digitalWrite(KILL_TOGGLE , HIGH);
  digitalWrite(LED_BUILTIN ,HIGH );
  LED_BUILTINSTATE = true;
}
void initComms(void)
{
  Serial.begin(9600);
  Serial.println("Power system status : " + powerStatus );
}




//=============SUBROUTINE/FUNCTIONAL CODES=======================
void updateComms(String message)
{
   Serial.println("Power system status : " + message );
}
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
