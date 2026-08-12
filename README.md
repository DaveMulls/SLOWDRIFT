# SLOWDRIFT
SLOWDRIFT is tape-inspired modulator, analog approximator, and colourizer.

## Getting started
Build the daisy libraries with:
```
make -C DaisySP
make -C libDaisy
```

Then flash your terrarium with:
```
cd your_pedal
# using USB (after entering bootloader mode)
make program-dfu
# using JTAG/SWD adaptor (like STLink)
make program
```

Note: The template pedal only turns the LED of the terrarium on and off and does no audio processing at all.
For an example with audio processing generated from this template you can checkout a [reverb](https://github.com/fxwiegand/terrarium-reverb).
