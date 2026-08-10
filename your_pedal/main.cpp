// SlowDrift - step 1 & 2: hardware layer + passthrough
// 48 kHz, block size 2. Audio is adc -> dac. No DSP yet.
//
// Controls are read in the MAIN LOOP, not the audio callback. This is the
// single biggest structural difference from the Heavy build, where 13
// parameters were dispatched into the DSP graph on every one of 24,000
// callbacks per second.

#include "daisy_seed.h"
#include "daisysp.h"

using namespace daisy;
using namespace daisysp;

// ---------------------------------------------------------------------------
// PIN MAPPING - verify this block against PedalPCB's terrarium.h before use.
// These follow the published Terrarium layout, but I have not verified them
// against hardware. Step 2 below (serial printing) is how you confirm them.
// ---------------------------------------------------------------------------
namespace pins
{
// Six pots, on ADC-capable pins
constexpr Pin KNOB[6] = {seed::A0, seed::A1, seed::A2,
                         seed::A3, seed::A4, seed::A5};

// Four SPDT toggle switches
constexpr Pin SWITCH[4] = {seed::D10, seed::D9, seed::D8, seed::D7};

// Two momentary footswitches
constexpr Pin FOOTSWITCH[2] = {seed::D25, seed::D26};

// Two indicator LEDs
constexpr Pin LED[2] = {seed::D22, seed::D23};
} // namespace pins

// ---------------------------------------------------------------------------
// Control state. Written by the main loop, read by the audio callback.
// Plain floats/bools: single-word reads and writes are atomic on Cortex-M7,
// so no locking is needed for this access pattern.
// ---------------------------------------------------------------------------
struct Controls
{
    float knob[6]  = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; // 0..1, smoothed
    bool  sw[4]    = {false, false, false, false};
    bool  fsHeld[2] = {false, false};
    bool  effectOn = false;
};

// ---------------------------------------------------------------------------
// DSP. Deliberately free of any libDaisy hardware types so this same struct
// compiles into a desktop WAV harness later - that is what makes iteration
// bearable. Right now it just passes audio through.
// ---------------------------------------------------------------------------
struct SlowDrift
{
    float sampleRate_ = 48000.f;

    void Init(float sampleRate) { sampleRate_ = sampleRate; }

    float Process(float in, const Controls& c)
    {
        if(!c.effectOn)
            return in;

        // Port the Pd signal chain here, one stage at a time:
        //   input filter -> saturation -> compressor -> tape delay + wow
        //   -> flavour bus -> tape stop -> output
        return in;
    }
};

// ---------------------------------------------------------------------------

DaisySeed     hw;
Controls      controls;
SlowDrift     slowdrift;
AnalogControl knobCtl[6];
Switch        swCtl[4];
Switch        fsCtl[2];
GPIO          ledOut[2];

void AudioCallback(AudioHandle::InterleavingInputBuffer  in,
                   AudioHandle::InterleavingOutputBuffer out,
                   size_t                                size)
{
    for(size_t i = 0; i < size; i += 2)
    {
        float s = slowdrift.Process(in[i], controls);
        out[i]     = s;
        out[i + 1] = s;
    }
}

int main(void)
{
    hw.Init();
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
    hw.SetAudioBlockSize(2);

    const float sr = hw.AudioSampleRate();
    slowdrift.Init(sr);

    // Knobs: one ADC config entry per pot, scanned by DMA in the background.
    AdcChannelConfig adcCfg[6];
    for(int i = 0; i < 6; i++)
        adcCfg[i].InitSingle(pins::KNOB[i]);
    hw.adc.Init(adcCfg, 6);
    hw.adc.Start();

    // 1 kHz update rate, light slew. Cheap because it runs in the main loop.
    for(int i = 0; i < 6; i++)
        knobCtl[i].Init(hw.adc.GetPtr(i), 1000.f, false, false, 0.01f);

    for(int i = 0; i < 4; i++)
        swCtl[i].Init(pins::SWITCH[i], 1000.f);
    for(int i = 0; i < 2; i++)
        fsCtl[i].Init(pins::FOOTSWITCH[i], 1000.f);
    for(int i = 0; i < 2; i++)
        ledOut[i].Init(pins::LED[i], GPIO::Mode::OUTPUT);

    hw.StartLog(false);
    hw.StartAudio(AudioCallback);

    uint32_t lastPrint = System::GetNow();

    for(;;)
    {
        for(int i = 0; i < 6; i++)
            controls.knob[i] = knobCtl[i].Process();

        for(int i = 0; i < 4; i++)
        {
            swCtl[i].Debounce();
            controls.sw[i] = swCtl[i].Pressed();
        }

        for(int i = 0; i < 2; i++)
        {
            fsCtl[i].Debounce();
            controls.fsHeld[i] = fsCtl[i].Pressed();
        }

        // Footswitch 1 latches the effect on and off.
        if(fsCtl[0].RisingEdge())
            controls.effectOn = !controls.effectOn;

        ledOut[0].Write(controls.effectOn);
        ledOut[1].Write(controls.fsHeld[1]);

        // Step 2 verification: confirm every control maps where you expect.
        // Delete this block once the mapping is confirmed.
        if(System::GetNow() - lastPrint > 250)
        {
            lastPrint = System::GetNow();
            hw.PrintLine("K %d %d %d %d %d %d | SW %d%d%d%d | FS %d%d | ON %d",
                         (int)(controls.knob[0] * 100),
                         (int)(controls.knob[1] * 100),
                         (int)(controls.knob[2] * 100),
                         (int)(controls.knob[3] * 100),
                         (int)(controls.knob[4] * 100),
                         (int)(controls.knob[5] * 100),
                         controls.sw[0],
                         controls.sw[1],
                         controls.sw[2],
                         controls.sw[3],
                         controls.fsHeld[0],
                         controls.fsHeld[1],
                         controls.effectOn);
        }

        System::Delay(1);
    }
}
