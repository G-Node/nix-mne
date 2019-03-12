randomize_trials=true;
write_codes = true;
pulse_width = 100;

begin;
text { caption="0"; font_color = 255,255,255; font_size = 400; }  tx0;
text { caption="1"; font_color = 255,255,255; font_size = 400;  } tx1;
text { caption="2"; font_color = 255,255,255; font_size = 400;  } tx2;
text { caption="3"; font_color = 255,255,255; font_size = 400; }  tx3;
text { caption="4"; font_color = 255,255,255; font_size = 400;  } tx4;
text { caption="5"; font_color = 255,255,255; font_size = 400;  } tx5;
text { caption="6"; font_color = 255,255,255; font_size = 400; }  tx6;
text { caption="7"; font_color = 255,255,255; font_size = 400;  } tx7;
text { caption="8"; font_color = 255,255,255; font_size = 400;  } tx8;
text { caption="9"; font_color = 255,255,255; font_size = 400; }  tx9;



  
 trial { 
    trial_duration=1000;
    start_delay=500;
    picture { text tx1; x=0; y=0;}; 
    port_code=1;
   
  } t1;
  
  trial {
    start_delay=500;
    trial_duration=1000; 
    picture {   text tx2; x=0; y=0; }; 
    port_code=2;
  } t2; 
  
  trial { 
    trial_duration=1000; 
    start_delay=500;
    picture {   text tx3; x=0; y=0; }; 
    port_code=3;
  } t3; 


trial { 
    trial_duration=1000;
    start_delay=500;
    picture { text tx4; x=0; y=0;}; 
    port_code=4;
  } t4;
  
  trial {
    start_delay=500;
    trial_duration=1000; 
    picture {   text tx5; x=0; y=0; }; 
   port_code=5;
  } t5; 
  
  trial { 
    trial_duration=1000; 
    start_delay=500;
    picture {   text tx6; x=0; y=0; }; 
    port_code=6;
  } t6;
  
trial { 
    trial_duration=1000;
    start_delay=500;
    picture { text tx7; x=0; y=0;}; 
   port_code=7;
  } t7;
  
  trial {
    start_delay=500;
    trial_duration=1000; 
    picture {   text tx8; x=0; y=0; }; 
   port_code=8;
  } t8; 
  
  trial { 
    trial_duration=1000; 
    start_delay=500;
    picture {   text tx9; x=0; y=0; }; 
    port_code=9;
  } t9; 


trial { 
    trial_duration=1000;
    start_delay=500;
    picture { text tx0; x=0; y=0;}; 
   port_code=10;
  } t0;
  
  
 
LOOP $i 50;  
  trial t1;
  trial t2;
  trial t3;
  trial t4;
  trial t5;
  trial t6;
  trial t7;
  trial t8;
  trial t9;
  trial t0;
  


ENDLOOP;  