// SlowDrift - full C++ port of the Pure Data patch (SD_LowLatency_v7.pd)
// 48 kHz, block size 2. Controls read in the main loop.
//
// The DSP primitives below deliberately reproduce Pure Data's exact
// coefficient maths (lop~, hip~, bp~, rpole~, vd~) so the voicing matches the
// patch rather than merely resembling it.
//
// CONTROLS
//   knob1  output level        0..4x
//   knob2  modulation depth    wow / vibrato / tremolo
//   knob3  modulation rate     + noise-wow amount
//   knob4  drive               1..31x into the saturator
//   knob5  flavour mode        6 positions
//   knob6  failure amount      dropouts, snags, bloom
//   sw1    dry blend on
//   sw2    compressor makeup
//   sw3    alt EQ (removes the pre-saturation dry path)
//   sw4    vibrato mode (replaces wow with a clean LFO)
//   fs1    effect on/off (latching). Hold 4 s alone to reset the alt layer.
//   fs2    tape stop (tap latches, hold is momentary)
//
//   ALT LAYER - hold both footswitches for 600 ms, both LEDs blink.
//   Each knob is inert until nudged, then takes over:
//     knob1  how much dry the dry switch passes
//     knob2  lag: centre delay the modulation swings around
//     knob3  modulation waveshape: sine / triangle / sample+hold / saw
//     knob4  low pass  - fully CW is off, turn back to roll off highs
//     knob5  mid        +/-18 dB at 775 Hz, flat at noon
//     knob6  high pass  - fully CCW is off, turn up to roll off bass

#include "daisy_seed.h"
#include "daisysp.h"
#include <cmath>

using namespace daisy;
using namespace daisysp;

// ===========================================================================
// Pd-faithful DSP primitives
// ===========================================================================

static inline float Clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

// heavylib hv.tanh~ equivalent: Pade rational approximation of tanh
static inline float FastTanh(float x)
{
    if(x < -3.f)
        return -1.f;
    if(x > 3.f)
        return 1.f;
    const float x2 = x * x;
    return x * (27.f + x2) / (27.f + 9.f * x2);
}

// Pd [lop~] : coef = clamp(2*pi*fc/sr, 0, 1); y += coef * (x - y)
struct PdLop
{
    float coef = 0.f, z = 0.f, sr = 48000.f;
    void  Init(float sampleRate, float fc)
    {
        sr = sampleRate;
        SetFreq(fc);
        z = 0.f;
    }
    void SetFreq(float fc) { coef = Clampf(2.f * (float)M_PI * fc / sr, 0.f, 1.f); }
    inline float Process(float x)
    {
        z += coef * (x - z);
        return z;
    }
};

// Pd [hip~] response, computed as x - lowpass(x).
// Mathematically the same one-pole high pass, but without the huge internal
// accumulator that costs precision in 32-bit float at low cutoffs.
struct PdHip
{
    PdLop  lp;
    void Init(float sr, float fc) { lp.Init(sr, fc); }
    void SetFreq(float fc) { lp.SetFreq(fc); }
    inline float Process(float x) { return x - lp.Process(x); }
};

// Pd [bp~] : two-pole resonator
struct PdBp
{
    float coef1 = 0.f, coef2 = 0.f, gain = 0.f;
    float last = 0.f, prev = 0.f, sr = 48000.f;

    static inline float QCos(float f)
    {
        if(f >= -1.5707963f && f <= 1.5707963f)
        {
            const float g = f * f;
            return (((g * g * g * (-1.f / 720.f) + g * g * (1.f / 24.f)) - g * 0.5f) + 1.f);
        }
        return 0.f;
    }
    void Init(float sampleRate, float f, float q)
    {
        sr   = sampleRate;
        last = prev = 0.f;
        Set(f, q);
    }
    void Set(float f, float q)
    {
        if(f < 0.001f)
            f = 10.f;
        if(q < 0.f)
            q = 0.f;
        const float omega      = f * (2.f * (float)M_PI) / sr;
        float       oneminusr  = (q < 0.001f) ? 1.f : omega / q;
        if(oneminusr > 1.f)
            oneminusr = 1.f;
        const float r = 1.f - oneminusr;
        coef1         = 2.f * QCos(omega) * r;
        coef2         = -r * r;
        gain          = 2.f * oneminusr * (oneminusr + r * omega);
    }
    inline float Process(float x)
    {
        const float output = x + coef1 * last + coef2 * prev;
        prev               = last;
        last               = output;
        return gain * output;
    }
};

// True two-pole bandpass (topology preserving state variable). Unlike Pd's
// [bp~], which is an all-pole resonator with a lowpass tail, this has a zero
// at DC - essential for a mid EQ band that is not to lift the bass with it.
struct SvfBand
{
    float ic1 = 0.f, ic2 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    void  Init(float sr, float fc, float q)
    {
        const float g = tanf(3.14159265f * fc / sr), k = 1.f / q;
        a1            = 1.f / (1.f + g * (g + k));
        a2            = g * a1;
        a3            = g * a2;
        ic1 = ic2 = 0.f;
    }
    inline float Process(float x)
    {
        const float v3 = x - ic2;
        const float v1 = a1 * ic1 + a2 * v3;
        const float v2 = ic2 + a2 * ic1 + a3 * v3;
        ic1            = 2.f * v1 - ic1;
        ic2            = 2.f * v2 - ic2;
        return v1; // bandpass, unity at fc
    }
};

// Pd [rpole~] : y[n] = x[n] + a * y[n-1]
struct PdRPole
{
    float y = 0.f;
    inline float Process(float x, float a)
    {
        y = x + a * y;
        return y;
    }
};

// Pd [line~] : linear ramp to a target over a time in ms
struct PdLine
{
    float value = 0.f, target = 0.f, inc = 0.f, sr = 48000.f;
    int   samplesLeft = 0;
    void  Init(float sampleRate, float v = 0.f)
    {
        sr = sampleRate;
        value = target = v;
        samplesLeft    = 0;
    }
    void Set(float t, float ms)
    {
        target = t;
        if(ms <= 0.f)
        {
            value       = t;
            samplesLeft = 0;
            inc         = 0.f;
            return;
        }
        samplesLeft = (int)(ms * 0.001f * sr);
        if(samplesLeft < 1)
            samplesLeft = 1;
        inc = (target - value) / (float)samplesLeft;
    }
    inline float Process()
    {
        if(samplesLeft > 0)
        {
            value += inc;
            if(--samplesLeft == 0)
                value = target;
        }
        return value;
    }
};

// Bipolar LFO shapes, all -1..+1 and phase-aligned with cosine so switching
// between them does not jump. Sample and hold is stateful and lives in
// ShapedLfo below rather than in this function.
enum LfoShape
{
    LFO_SINE = 0,
    LFO_TRI,
    LFO_SH,
    LFO_SAW,
    LFO_COUNT
};

static inline float LfoWave(float phase, int shape)
{
    switch(shape)
    {
        case LFO_TRI: return 4.f * fabsf(phase - 0.5f) - 1.f;
        // A true saw jumps from -1 straight back to +1, and an instantaneous
        // jump in delay time is a pop rather than a pitch move. This ramps
        // down over 88% of the cycle then flies back over the remaining 12%:
        // still unmistakably a sawtooth, but the return is a fast swoop.
        case LFO_SAW:
            return phase < 0.88f ? 1.f - 2.f * (phase / 0.88f)
                                 : -1.f + 2.f * ((phase - 0.88f) / 0.12f);
        default: return cosf(6.2831853f * phase);
    }
}

// Pd [phasor~] : 0..1 ramp
struct PdPhasor
{
    float phase = 0.f, inc = 0.f, sr = 48000.f;
    void  Init(float sampleRate) { sr = sampleRate; }
    void  SetFreq(float f) { inc = f / sr; }
    void  SetPhase(float p) { phase = p; }
    inline float Process()
    {
        phase += inc;
        while(phase >= 1.f)
            phase -= 1.f;
        while(phase < 0.f)
            phase += 1.f;
        return phase;
    }
};

// An LFO that owns its phase, its sample-and-hold state, and a slew stage.
// The slew is what the lag control drives: at zero it only rounds off the
// steps enough to keep them from clicking, wound up it turns every jump into
// a glide.
struct ShapedLfo
{
    PdPhasor ph;
    PdLop    lag;
    float    prevPhase = 0.f;
    float    shValue   = 0.f;
    uint32_t rng       = 1u;
    float    fcTop     = 60.f;

    void Init(float sr, uint32_t seed, float topCutoff)
    {
        ph.Init(sr);
        lag.Init(sr, topCutoff);
        rng   = seed;
        fcTop = topCutoff;
    }
    void SetFreq(float f) { ph.SetFreq(f); }

    // Advance the phase and return the shape, with no slew applied.
    inline float Raw(int shape)
    {
        const float p = ph.Process();
        if(p < prevPhase) // phase wrapped: latch a new random step
        {
            rng     = rng * 1664525u + 1013904223u;
            shValue = (float)(rng >> 9) * (1.f / 4194304.f) - 1.f;
        }
        prevPhase = p;

        return (shape == LFO_SH) ? shValue : LfoWave(p, shape);
    }

    // Sample and hold still steps instantly, so it gets a glide scaled to the
    // LFO period - about a tenth of a cycle. Slow settings stay percussive,
    // fast settings stop popping, and the character holds across the rate
    // range instead of being tuned for one speed. The continuous shapes only
    // need the light fixed slew.
    inline float Smoothed(float raw, int shape)
    {
        float fc = fcTop;
        if(shape == LFO_SH)
        {
            const float hz = ph.inc * ph.sr;
            const float f  = 1.6f * (hz > 0.05f ? hz : 0.05f);
            fc             = f < fcTop ? f : fcTop;
            if(fc < 1.2f)
                fc = 1.2f; // never so slow that the steps vanish entirely
        }
        lag.SetFreq(fc);
        return lag.Process(raw);
    }

    inline float Process(int shape) { return Smoothed(Raw(shape), shape); }
};

// Pd [delwrite~] / [vd~] with 4-point cubic interpolation, as vd~ uses.
// The buffer is supplied externally so the caller decides which RAM it lives in.
struct PdDelay
{
    float* buf = nullptr;
    size_t N   = 0;
    size_t wr  = 0;
    float  sr  = 48000.f;

    void Init(float sampleRate, float* buffer, size_t len)
    {
        sr  = sampleRate;
        buf = buffer;
        N   = len;
        for(size_t i = 0; i < N; i++)
            buf[i] = 0.f;
        wr = 0;
    }
    inline void Write(float x)
    {
        buf[wr] = x;
        wr      = (wr + 1) % N;
    }
    // Sample at delay k, in samples. wr points one past the newest sample.
    inline float At(int k) const
    {
        return buf[((int)wr - 1 - k + 2 * (int)N) % (int)N];
    }
    // Catmull-Rom between the samples at delay di and di+1.
    inline float ReadMs(float ms)
    {
        float d = ms * 0.001f * sr;
        if(d < 1.f)
            d = 1.f;
        if(d > (float)(N - 4))
            d = (float)(N - 4);
        const int   di = (int)d;
        const float t  = d - (float)di;
        const float p0 = At(di - 1);
        const float p1 = At(di);
        const float p2 = At(di + 1);
        const float p3 = At(di + 2);
        return 0.5f
               * ((2.f * p1) + (-p0 + p2) * t
                  + (2.f * p0 - 5.f * p1 + 4.f * p2 - p3) * t * t
                  + (-p0 + 3.f * p1 - 3.f * p2 + p3) * t * t * t);
    }
};

// Buffer lengths at 48 kHz
enum : size_t
{
    SD_TAPE_LEN    = 9600,  //  200 ms, headroom for the lag offset
    SD_FLANGE_LEN  = 800,   //   15 ms
    SD_TSTAPE_LEN  = 29000, //  600 ms - the big one, put this in SDRAM
    SD_TSPITCH_LEN = 6200,  //  128 ms
    SD_SLAP_LEN    = 7400   //  150 ms
};

// Pd [noise~]
struct PdNoise
{
    uint32_t state = 22222;
    inline float Process()
    {
        state = state * 1664525u + 1013904223u;
        return (float)((int32_t)state) * 4.6566129e-10f;
    }
};

// ===========================================================================
// Control block, written by the main loop
// ===========================================================================
struct Controls
{
    float knob[6]  = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    // Values that live behind the alt layer. These are latched, not read
    // straight off a knob, so the primary control keeps its position.
    float volume    = 0.5f; // knob 1 normally
    float dryAmount = 0.5f; // knob 1 while both footswitches are held
    float rate      = 0.3f; // knob 3 normally
    int   lfoShape  = LFO_SINE; // knob 3 while both footswitches are held
    float wowDepth  = 0.3f; // knob 2 normally
    float lag       = 0.f;  // knob 2 on the alt layer: centre delay time
    float sat       = 0.3f; // knob 4 normally
    float flavour   = 0.f;  // knob 5 normally
    float fail      = 0.f;  // knob 6 normally
    float eqLow     = 1.0f; // knob 4 on the alt layer, 1.0 = no low pass
    float eqMid     = 0.5f; // knob 5 on the alt layer, 0.5 = flat
    float eqHigh    = 0.0f; // knob 6 on the alt layer, 0.0 = no high pass
    bool  dryOn    = false; // sw1
    bool  compOn   = false; // sw2
    bool  bloomOn  = false; // sw3
    bool  vibrato  = false; // sw4
    bool  effectOn = false; // fs1, latching
    bool  tapeStop = false; // fs2: tap latches, hold is momentary
};

// ===========================================================================
// The pedal. No libDaisy types, so this also builds in a desktop WAV harness.
// ===========================================================================
// Restores unity after the two 0.5 stages between the tape delay and the bus.
static constexpr float kMakeup = 5.5f;

struct SlowDrift
{
    float sr = 48000.f;

    // --- input / output conditioning
    PdHip inHp, outHp, preHp50, satHp1, satHp2;
    PdLop eqLp1, eqLp2;
    // Alt-layer values arrive stepped at control rate. Fed straight into a
    // delay time or a filter cutoff, each step is an instantaneous jump - a
    // click - and turning the knob produces a stream of them. Smooth them all.
    PdLop lagSm, drySm, eqLowSm, eqMidSm, eqHighSm;
    PdHip eqHp1, eqHp2;
    SvfBand eqMidBp;

    // --- bloom (lpg_engine)
    PdHip   lpgHp;
    PdLop   lpgAtk, lpgRel, lpgRipple, lpgKnob;
    float   lpgOpen = 0.f;
    PdRPole lpgP1, lpgP2;

    // --- saturation and compressor
    PdLop   driveSm, satTilt;
    float compEnvSq = 0.f;
    float compGain  = 1.f;

    // --- tape delay and wow
    PdDelay tape;
    PdNoise       noise;
    PdLop         wowN1, wowN2, wowN3;   // noise~ -> lop 0.8 / 15 / 8
    PdLine        walkLine;              // random-walk target ramp
    PdLop         walkS1, walkS2, walkHp;
    PdPhasor      sine1, sine2, sine3;   // 0.61 / 1.07 / 1.83 Hz
    ShapedLfo     vibLfo, wowLfo;
    PdLop         vibDepth, altDepth, snagSm1, snagSm2;
    float       walkTarget = 0.f;
    int         walkTimer  = 0;
    float       snagTime   = 2.f;
    int         snagTimer  = -1;

    // --- failure engine
    int   failTimer     = 0;
    int   dropTimer     = 0;
    float failAmt       = 0.f;  // latched: knob6 when the bloom switch is OFF
    float bloomAmt      = 0.f;  // latched: knob6 when the bloom switch is ON
    float dropoutTarget = 1.f;
    PdLop   dropoutSm;

    // --- flavour bus
    PdLine  modeGain[6];
    PdDelay slap;
    PdLop   slapLp;
    int   currentMode = -1;
    // mode 1: harmonic
    // Knobs filter: topology-preserving 2-pole state variable low pass
    float kf1 = 0.f, kf2 = 0.f, kfA1 = 0.f, kfA2 = 0.f, kfA3 = 0.f;
    float kg1 = 0.f, kg2 = 0.f, kgA1 = 0.f, kgA2 = 0.f, kgA3 = 0.f;
    // mode 2: alt EQ
    PdHip   f2h1, f2h2;
    PdLop   f2l1, f2l2;
    PdBp    f2bp;
    // mode 3: tape flange
    PdDelay flange;
    PdLop         flangeMod1, flangeMod2;
    // mode 4: tremolo
    ShapedLfo tremLfo;
    PdLop    tremDepth;
    // mode 5: slapback (declared with the flavour members above)

    // --- tape stop
    PdDelay tsTape;
    PdDelay tsPitch;
    PdLine         tsRate, tsTapeGain;
    float          tsDelayMs = 2.f;
    bool           tsResetPending = false;
    PdPhasor       tsGrain1, tsGrain2;
    bool         tsPrev = false;
    int          tsStage = -1, tsTimer = 0;

    uint32_t rng = 12345;
    inline float Rand01()
    {
        rng = rng * 1664525u + 1013904223u;
        return (float)(rng >> 8) * (1.f / 16777216.f);
    }

    void Init(float sampleRate,
              float* bTape,
              float* bFlange,
              float* bTsTape,
              float* bTsPitch,
              float* bSlap)
    {
        sr = sampleRate;

        inHp.Init(sr, 10.f);
        outHp.Init(sr, 10.f);
        preHp50.Init(sr, 50.f);
        lagSm.Init(sr, 4.f);
        drySm.Init(sr, 8.f);
        eqLowSm.Init(sr, 6.f);
        eqMidSm.Init(sr, 6.f);
        eqHighSm.Init(sr, 6.f);
        eqLp1.Init(sr, 20000.f);
        eqLp2.Init(sr, 20000.f);
        eqHp1.Init(sr, 20.f);
        eqHp2.Init(sr, 20.f);
        eqMidBp.Init(sr, 775.f, 0.85f);
        satHp1.Init(sr, 30.f);
        satHp2.Init(sr, 10.f);

        lpgHp.Init(sr, 10.f);
        lpgAtk.Init(sr, 80.f);
        lpgRel.Init(sr, 3.f);
        lpgRipple.Init(sr, 20.f);
        lpgKnob.Init(sr, 10.f);

        driveSm.Init(sr, 5.f);
        satTilt.Init(sr, 12000.f);

        tape.Init(sr, bTape, SD_TAPE_LEN);
        wowN1.Init(sr, 0.8f);
        wowN2.Init(sr, 15.f);
        wowN3.Init(sr, 8.f);
        walkLine.Init(sr, 0.f);
        walkS1.Init(sr, 0.325f);
        walkS2.Init(sr, 0.325f);
        walkHp.Init(sr, 0.195f);
        sine1.Init(sr);
        sine1.SetFreq(0.397f);
        sine2.Init(sr);
        sine2.SetFreq(0.696f);
        sine2.SetPhase(0.37f);
        sine3.Init(sr);
        sine3.SetFreq(1.19f);
        sine3.SetPhase(0.71f);
        vibLfo.Init(sr, 0x9E3779B9u, 60.f);
        wowLfo.Init(sr, 0xC2B2AE35u, 40.f);
        vibDepth.Init(sr, 10.f);
        altDepth.Init(sr, 20.f);
        snagSm1.Init(sr, 1.f);
        snagSm2.Init(sr, 1.f);
        dropoutSm.Init(sr, 10.f);

        for(int i = 0; i < 6; i++)
            modeGain[i].Init(sr, i == 0 ? 1.f : 0.f);
        slap.Init(sr, bSlap, SD_SLAP_LEN);
        slapLp.Init(sr, 4000.f);

        {
            const float g = tanf(3.14159265f * 3600.f / sr), k = 1.f / 1.7f;
            kfA1          = 1.f / (1.f + g * (g + k));
            kfA2          = g * kfA1;
            kfA3          = g * kfA2;
        }
        {
            const float g = tanf(3.14159265f * 4200.f / sr), k = 1.f / 0.707f;
            kgA1          = 1.f / (1.f + g * (g + k));
            kgA2          = g * kgA1;
            kgA3          = g * kgA2;
        }
        f2h1.Init(sr, 500.f);
        f2h2.Init(sr, 500.f);
        f2l1.Init(sr, 1800.f);
        f2l2.Init(sr, 1800.f);
        f2bp.Init(sr, 1000.f, 0.6f); // wider, less peaky
        flange.Init(sr, bFlange, SD_FLANGE_LEN);
        flangeMod1.Init(sr, 0.3f);
        flangeMod2.Init(sr, 0.3f);
        tremLfo.Init(sr, 0x85EBCA77u, 500.f);
        tremDepth.Init(sr, 10.f);

        tsTape.Init(sr, bTsTape, SD_TSTAPE_LEN);
        tsPitch.Init(sr, bTsPitch, SD_TSPITCH_LEN);
        tsRate.Init(sr, 1.f);
        tsTapeGain.Init(sr, 1.f);
        tsDelayMs = 2.f;
        tsGrain1.Init(sr);
        tsGrain2.Init(sr);
        tsGrain2.SetPhase(0.5f);
    }

    // Control-rate housekeeping. Called once per audio block, not per sample.
    void UpdateControls(const Controls& c, int blockSize)
    {
        // knob5 selects one of six flavour modes with a 0.5 ms crossfade
        int mode = (int)(c.flavour * 6.f);
        if(mode > 5)
            mode = 5;
        if(mode != currentMode)
        {
            currentMode = mode;
            for(int i = 0; i < 6; i++)
                modeGain[i].Set(i == mode ? 1.f : 0.f, 0.5f);
        }

        // LFO rates follow knob3
        vibLfo.SetFreq(0.15f + c.rate * 6.8f);
        wowLfo.SetFreq(0.25f + c.rate * 2.5f); // wow rates, slower than vibrato
        tremLfo.SetFreq(0.5f + c.rate * 14.5f);

        // knob6 drives either bloom or failure depending on the bloom switch.
        // The inactive one holds its last position rather than following along.
        if(c.bloomOn)
            bloomAmt = c.fail;
        else
            failAmt = c.fail;

        // --- failure engine ----------------------------------------------
        // metro period 50..500 ms, shorter as the knob rises
        const float periodMs = 50.f + (1.f - failAmt) * 450.f;
        failTimer -= blockSize;
        if(failTimer <= 0)
        {
            failTimer = (int)(periodMs * 0.001f * sr);
            if(failAmt > 0.001f)
            {
                const float roll = Rand01();
                if(roll < failAmt * 0.9f)
                {
                    dropoutTarget = 0.1f;
                    dropTimer     = (int)(0.080f * sr); // Pd restores after 80 ms
                }
                if(roll < failAmt * 0.7f && snagTimer < 0)
                    snagTimer = (int)(0.300f * sr); // snag in 300 ms
            }
        }
        if(dropTimer > 0 && (dropTimer -= blockSize) <= 0)
        {
            dropoutTarget = 1.f;
            dropTimer     = 0;
        }
        if(snagTimer >= 0 && (snagTimer -= blockSize) <= 0)
        {
            snagTime  = 2.f + failAmt * 20.f;
            snagTimer = -(int)(0.200f * sr);
        }
        else if(snagTimer < -1)
        {
            snagTimer += blockSize;
            if(snagTimer >= -1)
            {
                snagTimer = -1;
                snagTime  = 2.f;
            }
        }

        // --- tape stop -----------------------------------------------------
        // Playback rate is what we control; the delay is its integral.
        //   pitch ratio = 1 - d(delay)/dt
        // Slowing down therefore means the delay must GROW, and speeding back
        // up means it must keep growing, only more slowly. Sweeping the delay
        // back down is what made the restart pitch up instead of down.
        if(c.tapeStop != tsPrev)
        {
            tsPrev = c.tapeStop;
            if(c.tapeStop)
            {
                tsRate.Set(0.f, 900.f);     // decelerate to a halt
                tsTapeGain.Set(0.f, 1100.f);
                tsStage = 0;
            }
            else
            {
                // Brief mute so the read head can be reset without a click.
                tsTapeGain.Set(0.f, 15.f);
                tsStage = 3;
                tsTimer = (int)(0.015f * sr);
            }
        }
        if(tsStage == 3 && (tsTimer -= blockSize) <= 0)
        {
            // Reset to minimum so the spin-up has room to grow. Without this
            // the delay sat clamped at the ceiling, d(delay)/dt was zero, and
            // the restart produced no pitch effect at all.
            tsResetPending = true;
            tsRate.Set(1.f, 500.f); // accelerate from a standstill: pitch rises
            tsTapeGain.Set(1.f, 260.f);
            tsStage = 1;
            tsTimer = (int)(0.500f * sr);
        }
        else if(tsStage == 1 && (tsTimer -= blockSize) <= 0)
        {
            // Spin-up leaves ~200 ms of accumulated lag. Bleeding it off as a
            // pitch drift left the pedal laggy for seconds, so instead the
            // delayed voice is crossfaded out, the read head reset, and the
            // voice faded back in. 90 ms total and the delay is gone.
            tsRate.Set(1.f, 5.f);
            tsTapeGain.Set(0.f, 45.f);
            tsStage = 2;
            tsTimer = (int)(0.045f * sr);
        }
        else if(tsStage == 2 && (tsTimer -= blockSize) <= 0)
        {
            tsResetPending = true;
            tsTapeGain.Set(1.f, 45.f);
            tsStage = -1;
        }
    }

    inline float Process(float in, const Controls& c)
    {
        const float dry = inHp.Process(in);

        if(!c.effectOn)
            return Clampf(dry, -0.99f, 0.99f);

        // ---------------- bloom: low pass gate, after the Shallow Water ---
        // Fairfield's manual: "the input's envelope is followed and used to
        // modulate the frequency of a low pass filter. The amount of envelope
        // is set by the LPG control." Low settings are dark, choke subtle
        // notes and cut sustain; high settings are bright and bouncy.
        //
        // The envelope is deliberately asymmetric - a 2 ms attack follower and
        // a 53 ms release follower, combined with max(). The fast one snaps
        // the gate open on the pick, the slow one lets it fall away
        // afterwards, and that difference IS the bloom.
        const float lpgIn  = lpgHp.Process(dry);
        const float rect   = fabsf(lpgIn);
        const float atk    = lpgAtk.Process(rect);   // lop~ 80, ~2 ms
        const float rel    = lpgRel.Process(rect);   // lop~ 3,  ~53 ms
        const float envRaw = lpgRipple.Process(atk > rel ? atk : rel);
        // Sensitivity 6, as in the Pd patch. Raising it to 18 pinned the
        // envelope at full scale two thirds of the time, which left the gate
        // permanently open and made the whole thing a static low pass.
        const float env  = Clampf(envRaw * 6.f, 0.f, 1.f);
        const float knob = lpgKnob.Process(bloomAmt);
        // knob * env is the swinging part; knob^2 is a resting floor, so that
        // fully clockwise pins the gate open and fully anticlockwise closes it
        // down to a 190 Hz whisper.
        lpgOpen = Clampf(env * knob + knob * knob, 0.f, 1.f);

        // ---------------- pre-saturation sum -----------------------------
        // bloom is summed with the dry path, then high-passed at 50 Hz
        // With bloom engaged the LPG voice REPLACES the dry path - in the Pd
        // patch that is what the alt_eq switch did, and summing the dry in
        // permanently is why the gate sounded like a weak static low pass.
        const float x = preHp50.Process(dry);

        // ---------------- drive + saturation -----------------------------
        const float drive = driveSm.Process(1.f + c.sat * 24.f);
        float pre = x * drive * 0.5f;
        pre += 0.06f * pre * pre;                 // slight asymmetry, 2nd harmonic
        // Cubic soft clip: essentially linear at low level, so the knob is
        // genuinely clean at zero, then thickens progressively.
        float sat = (pre > 1.f) ? 1.f
                                : ((pre < -1.f) ? -1.f : 1.5f * (pre - pre * pre * pre / 3.f));
        sat /= 0.75f * sqrtf(drive);              // level compensation
        // Tape loses top end as it saturates: 12 kHz down to about 3.5 kHz.
        satTilt.SetFreq(12000.f - c.sat * 8500.f);
        sat = satTilt.Process(sat);
        sat = satHp1.Process(sat);

        // ---------------- compressor (env~ 256 style, ratio 80) ----------
        compEnvSq += (sat * sat - compEnvSq) * 0.004f; // ~256-sample window
        const float rms  = sqrtf(compEnvSq) + 1e-9f;
        // heavylib's threshold for [hv.compressor~ 80] could not be determined
        // from the patch, so this is a judgement call: 4:1 above -10 dBFS.
        // Hard limiting flattened the saturation knob to inaudibility.
        // Switch off: gentle 4:1 glue. Switch on: FET-style, low threshold,
        // 10:1, fast attack and slow release, with makeup to match - obvious
        // pumping rather than a level change.
        const float thr    = c.compOn ? 0.06f : 0.3f;
        const float ratio  = c.compOn ? 10.f : 4.f;
        float       want   = 1.f;
        if(rms > thr)
            want = powf(thr / rms, 1.f - 1.f / ratio);
        const float coefUp = c.compOn ? 0.0035f : 0.001f;  // attack
        const float coefDn = c.compOn ? 0.00012f : 0.001f; // release
        compGain += (want - compGain) * (want < compGain ? coefUp : coefDn);
        float comp = sat * compGain * kMakeup * (c.compOn ? 3.2f : 1.f);
        comp       = FastTanh(comp);
        comp       = satHp2.Process(comp);

        // ---------------- wow / vibrato modulation -----------------------
        // noise branch
        float n = wowN3.Process(wowN2.Process(wowN1.Process(noise.Process()))) * 1000.f;
        // clamp the excursion to the pedestal so the sum stays positive and
        // smooth instead of slamming into the 1 ms floor
        const float ped = c.rate * 6.f;
        n               = Clampf(n, -ped, 18.f);
        const float k3   = c.rate / 3.f;
        const float wowN = (n + c.rate * 6.f) * k3;

        // random-walk branch
        if(--walkTimer <= 0)
        {
            walkTimer = (int)((0.769f + Rand01() * 1.538f) * sr);
            walkTarget = Rand01() * 2.f - 1.f;
            walkLine.Set(walkTarget, 1231.f + c.wowDepth * 1846.f);
        }
        float w = walkS2.Process(walkS1.Process(walkLine.Process()));
        w       = (w - walkHp.Process(w)) * 1.05f;

        // sine trio
        const float s = (cosf(6.2831853f * sine1.Process())
                         + cosf(6.2831853f * sine2.Process())
                         + cosf(6.2831853f * sine3.Process()))
                        * 0.221f;

        const float depth = altDepth.Process(c.wowDepth);

        // At the sine setting the wow is the tuned tape modulator: three slow
        // LFOs plus the high-passed random walk. Any other shape swaps in a
        // single shaped LFO so triangle, sample-and-hold and saw actually
        // reach the tape delay. Lag applies either way - previously it only
        // touched the vibrato and tremolo LFOs, which is why it did nothing
        // unless one of those was engaged.
        const float rawMod = (c.lfoShape == LFO_SINE)
                                 ? Clampf(w + s, -1.f, 1.f)
                                 : wowLfo.Raw(c.lfoShape);
        const float mod    = Clampf(wowLfo.Smoothed(rawMod, c.lfoShape), -1.f, 1.f);
        const float wowA  = mod * depth * 5.f + depth * 5.f;

        // vibrato branch (replaces wow when sw4 is up)
        const float vdep = vibDepth.Process(c.wowDepth * 3.f);
        // Sample and hold and saw step the delay time instantly, which reads
        // as a click rather than a pitch jump, so this LFO always carries some
        // slew even with lag at zero. The lag knob extends it into a glide.
        const float vib = vibLfo.Process(c.lfoShape) * vdep;

        const float snag = snagSm2.Process(snagSm1.Process(snagTime));

        // LAG, after the Walrus Julia: it sets the centre delay time the LFO
        // modulates around. On a bucket brigade a longer delay means a slower
        // clock, so the same clock wobble swings the time further - which is
        // why the Julia goes from tight and bright to seasick detune as you
        // open it. Both the offset and the swing scale together here.
        const float lagS     = lagSm.Process(c.lag);
        const float lagMs    = lagS * 38.f;
        const float lagScale = 1.f + lagS * 3.f;

        float delayMs;
        if(c.vibrato)
            delayMs = 4.f + lagMs + vib * lagScale + snag;
        else
            delayMs = lagMs + (wowN + wowA) * lagScale + snag;
        delayMs = Clampf(delayMs, 1.f, 190.f);

        tape.Write(comp);
        float t = tape.ReadMs(delayMs) * 0.5f;

        // ---------------- dropout ---------------------------------------
        t *= dropoutSm.Process(dropoutTarget);
        t *= 0.5f;

        // ---------------- flavour bus ------------------------------------
        const float f = t;
        float       bus = 0.f;

        const float g0 = modeGain[0].Process();
        const float g1 = modeGain[1].Process();
        const float g2 = modeGain[2].Process();
        const float g3 = modeGain[3].Process();
        const float g4 = modeGain[4].Process();
        const float g5 = modeGain[5].Process();

        // 0 - clean
        bus += f * g0;

        // 1 - the Knobs filter. Two-pole low pass at 3 kHz with mild
        //     resonance: keeps the 2-4 kHz harmonics that carry pick detail,
        //     drops the fizz and string noise above 6 kHz. Rounded, not dark.
        {
            // Stage 1: resonant peak at 3.6 kHz keeps pick detail present.
            float v3 = f - kf2;
            float v1 = kfA1 * kf1 + kfA2 * v3;
            float v2 = kf2 + kfA2 * kf1 + kfA3 * v3;
            kf1      = 2.f * v1 - kf1;
            kf2      = 2.f * v2 - kf2;
            // Stage 2: Butterworth at 4.2 kHz, taking the slope to 24 dB/oct
            // so everything above the peak falls away quickly.
            v3       = v2 - kg2;
            v1       = kgA1 * kg1 + kgA2 * v3;
            float w2 = kg2 + kgA2 * kg1 + kgA3 * v3;
            kg1      = 2.f * v1 - kg1;
            kg2      = 2.f * w2 - kg2;
            bus += w2 * 1.0f * g1;
        }

        // 2 - alt EQ
        {
            const float band = f2l2.Process(f2l1.Process(f2h2.Process(f2h1.Process(f))));
            const float peak = f2bp.Process(band) * 2.5f; // was 6, too harsh
            bus += (band + peak) * 0.55f * g2;
        }

        // 3 - tape flange
        {
            const float mt = flangeMod2.Process(flangeMod1.Process(noise.Process()));
            const float fd = 7.f + Clampf(mt * 600.f, -5.f, 5.f);
            const float fv = flange.ReadMs(Clampf(fd, 1.f, 14.f));
            flange.Write(f + fv * 0.2f);
            bus += (f + fv) * 0.6f * g3;
        }

        // 4 - tremolo
        {
            const float td = tremDepth.Process(c.wowDepth * 0.5f);
            const float lf = tremLfo.Process(c.lfoShape) * td + (1.f - td);
            bus += f * lf * g4;
        }

        // 5 - slapback
        {
            slap.Write(f);
            const float sv = slapLp.Process(slap.ReadMs(120.f)) * 0.5f;
            bus += (f + sv) * g5;
        }

        // ---------------- bloom gate -------------------------------------
        // Shallow Water's block diagram puts the voltage controlled low pass
        // after the delay line and before the dry/wet mix, so that is where it
        // sits: it gates the modulated voice, not the input, and the dry blend
        // stays clean. Two one-poles for 12 dB/oct, plus a VCA on the same
        // control - brightness and loudness move together, which is what makes
        // an LPG sound different from a plain filter.
        if(c.bloomOn)
        {
            const float o2     = lpgOpen * lpgOpen; // vactrol-ish curve
            const float cutoff = o2 * 0.925f + 0.025f;
            const float fb     = 1.f - cutoff;
            float       g      = lpgP1.Process(bus * cutoff, fb);
            g                  = lpgP2.Process(g * cutoff, fb);
            bus                = g * (lpgOpen * 0.65f + 0.35f);
        }

        // ---------------- dry blend --------------------------------------
        // Pd sums this into the bus ahead of the tape stop, so it is subject
        // to the tape-stop envelope and the output level knob like everything
        // else. Adding it after the level control made it far too loud.
        if(c.dryOn)
            bus += comp * drySm.Process(c.dryAmount) * 0.5f;

        // ---------------- tape stop --------------------------------------
        tsTape.Write(bus);
        tsPitch.Write(bus);

        if(tsResetPending)
        {
            tsDelayMs      = 2.f;
            tsResetPending = false;
        }
        const float rate = tsRate.Process();
        tsDelayMs += (1.f - rate) * 1000.f / sr;
        tsDelayMs = Clampf(tsDelayMs, 2.f, 560.f);
        float stopped = tsTape.ReadMs(tsDelayMs) * tsTapeGain.Process();

        // ---------------- output -----------------------------------------
        // ---------------- EQ, after the Condor -----------------------------
        // Bass +/-12 dB shelf, parametric-style mid +/-18 dB at 775 Hz (the
        // geometric centre of the Condor's 150 Hz - 4 kHz mid sweep), treble
        // +/-12 dB shelf. All flat at noon.
        // LOW  knob: low pass. Fully clockwise is transparent; turning back
        //            rolls the top off, like a tone control.
        // MID  knob: +/-18 dB at 775 Hz, flat at noon.
        // HIGH knob: high pass. Fully counter-clockwise is transparent;
        //            turning up rolls the bass away.
        const float eL = eqLowSm.Process(c.eqLow);
        const float eM = eqMidSm.Process(c.eqMid);
        const float eH = eqHighSm.Process(c.eqHigh);

        float eqd = stopped;
        if(eM < 0.499f || eM > 0.501f)
        {
            const float gM = powf(8.f, (eM - 0.5f) * 2.f);
            eqd += (gM - 1.f) * eqMidBp.Process(stopped);
        }
        if(eL < 0.99f)
        {
            const float fc = 250.f * powf(80.f, eL); // 250 Hz .. 20 kHz
            eqLp1.SetFreq(fc);
            eqLp2.SetFreq(fc);
            eqd = eqLp2.Process(eqLp1.Process(eqd));
        }
        if(eH > 0.01f)
        {
            const float fc = 20.f * powf(75.f, eH); // 20 Hz .. 1.5 kHz
            eqHp1.SetFreq(fc);
            eqHp2.SetFreq(fc);
            eqd = eqHp2.Process(eqHp1.Process(eqd));
        }

        float outv = eqd * (c.volume * 2.f); // noon = unity
        outv = outHp.Process(outv);
        return Clampf(outv, -0.99f, 0.99f);
    }
};

// ===========================================================================
// Hardware
// ===========================================================================
namespace pins
{
// Terrarium knobs sit on A1..A6. Reading A0 (unconnected) made the level
// control float, which is what was driving the output into distortion.
constexpr Pin KNOB[6] = {seed::A1, seed::A2, seed::A3,
                         seed::A4, seed::A5, seed::A6};
constexpr Pin SWITCH[4]     = {seed::D10, seed::D9, seed::D8, seed::D7};
constexpr Pin FOOTSWITCH[2] = {seed::D25, seed::D26};
constexpr Pin LED[2]        = {seed::D22, seed::D23};
} // namespace pins

// The 600 ms tape-stop buffer is 113 KB and will not fit in DTCMRAM alongside
// the others, so it lives in the Seed's external SDRAM. The rest total 56 KB
// and stay in fast internal RAM.
static float DSY_SDRAM_BSS bufTsTape[SD_TSTAPE_LEN];
static float               bufTape[SD_TAPE_LEN];
static float               bufFlange[SD_FLANGE_LEN];
static float               bufTsPitch[SD_TSPITCH_LEN];
static float               bufSlap[SD_SLAP_LEN];

// How far knob 1 must turn before it takes over an alt-layer parameter.
// 0.03 is about 8 degrees of a 270-degree pot: clearly deliberate, and far
// above the residual ADC noise left after AnalogControl's smoothing.
static constexpr float kAltDeadband = 0.03f;

float         altEntry[6]       = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
bool          altArmed[6]       = {false, false, false, false, false, false};
// Armed at boot so the knobs are live from power-up; disarmed again each time
// the alt layer is left.
float         normEntry[6]      = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
bool          normArmed[6]      = {true, true, true, true, true, true};
uint32_t      fs1Down           = 0;
uint32_t      flashUntil        = 0;
uint32_t      flashStart        = 0;
bool          holdRestored      = false;
bool          resetDone         = false;
uint32_t      fs2Down           = 0;
bool          tapeLatch         = false;
uint32_t      bothSince         = 0;
bool          altMode           = false;
bool          comboUsed         = false;
bool          effectBeforeCombo = false;

DaisySeed     hw;
Controls      controls;
SlowDrift     pedal;
AnalogControl knobCtl[6];
Switch        swCtl[4];
Switch        fsCtl[2];
GPIO          ledOut[2];

void AudioCallback(AudioHandle::InterleavingInputBuffer  in,
                   AudioHandle::InterleavingOutputBuffer out,
                   size_t                                size)
{
    pedal.UpdateControls(controls, (int)size / 2);
    for(size_t i = 0; i < size; i += 2)
    {
        const float s = pedal.Process(in[i], controls);
        out[i]        = s;
        out[i + 1]    = s;
    }
}

int main(void)
{
    hw.Init();
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
    hw.SetAudioBlockSize(2);

    const float sr = hw.AudioSampleRate();
    pedal.Init(sr, bufTape, bufFlange, bufTsTape, bufTsPitch, bufSlap);

    AdcChannelConfig adcCfg[6];
    for(int i = 0; i < 6; i++)
        adcCfg[i].InitSingle(pins::KNOB[i]);
    hw.adc.Init(adcCfg, 6);
    hw.adc.Start();

    for(int i = 0; i < 6; i++)
        knobCtl[i].Init(hw.adc.GetPtr(i), 1000.f, false, false, 0.01f);
    for(int i = 0; i < 4; i++)
        swCtl[i].Init(pins::SWITCH[i], 1000.f);
    for(int i = 0; i < 2; i++)
        fsCtl[i].Init(pins::FOOTSWITCH[i], 1000.f);
    for(int i = 0; i < 2; i++)
        ledOut[i].Init(pins::LED[i], GPIO::Mode::OUTPUT);

    hw.StartAudio(AudioCallback);

    for(;;)
    {
        for(int i = 0; i < 6; i++)
            controls.knob[i] = knobCtl[i].Process();

        for(int i = 0; i < 4; i++)
            swCtl[i].Debounce();
        controls.dryOn   = swCtl[0].Pressed();
        controls.compOn  = swCtl[1].Pressed();
        controls.bloomOn = swCtl[2].Pressed();
        controls.vibrato = swCtl[3].Pressed();

        for(int i = 0; i < 2; i++)
            fsCtl[i].Debounce();

        const uint32_t now      = System::GetNow();
        const bool     bothDown = fsCtl[0].Pressed() && fsCtl[1].Pressed();

        // ---- entering the alt layer -------------------------------------
        // Hold both switches for 600 ms. The individual presses will already
        // have toggled the effect and armed the tape stop, so those are undone
        // on entry rather than made to feel laggy by waiting to act.
        if(bothDown)
        {
            if(bothSince == 0)
                bothSince = now;
            else if(!altMode && now - bothSince > 600)
            {
                altMode = true;
                for(int i = 0; i < 6; i++)
                {
                    altEntry[i] = controls.knob[i];
                    altArmed[i] = false;
                }
                controls.effectOn = effectBeforeCombo;
                tapeLatch         = false;
                comboUsed         = true;
            }
        }
        else
        {
            bothSince = 0;
            if(altMode && !fsCtl[0].Pressed() && !fsCtl[1].Pressed())
            {
                altMode = false;
                // Same deal on the way out: whatever was dialled in before the
                // alt layer stays put until a knob is actually turned, so
                // visiting the alt layer never disturbs the main settings.
                for(int i = 0; i < 6; i++)
                {
                    normEntry[i] = controls.knob[i];
                    normArmed[i] = false;
                }
            }
        }

        // ---- footswitch 1: effect on/off, or hold 5 s to reset ----------
        if(fsCtl[0].RisingEdge())
        {
            fs1Down           = now;
            resetDone         = false;
            holdRestored      = false;
            effectBeforeCombo = controls.effectOn;
            controls.effectOn = !controls.effectOn;
        }
        // A press toggles the effect immediately, which is what a footswitch
        // has to do. But once it has been held past 600 ms it is clearly not a
        // stomp, so the toggle is undone right there - long before the reset
        // fires - and holding to reset leaves the pedal exactly as it was.
        if(fsCtl[0].Pressed() && !holdRestored && now - fs1Down > 600)
        {
            controls.effectOn = effectBeforeCombo;
            holdRestored      = true;
        }
        if(fsCtl[0].Pressed() && !fsCtl[1].Pressed() && !resetDone
           && now - fs1Down > 4000)
        {
            // Everything on the alt layer back to neutral. The effect toggle
            // caused by the initial press is undone as well, so a reset does
            // not leave the pedal switched.
            controls.dryAmount = 0.5f;
            controls.lag       = 0.f;
            controls.lfoShape  = LFO_SINE;
            controls.eqLow     = 1.0f; // low pass off
            controls.eqMid     = 0.5f; // flat
            controls.eqHigh    = 0.0f; // high pass off
            resetDone          = true;
            flashStart         = now;
            flashUntil         = now + 900; // three 150 ms flashes
        }

        // ---- footswitch 2: tap latches, hold is momentary ---------------
        if(fsCtl[1].RisingEdge())
            fs2Down = now;
        if(fsCtl[1].FallingEdge())
        {
            if(comboUsed)
                comboUsed = false; // this release was part of the combo
            else if(now - fs2Down < 400)
                tapeLatch = !tapeLatch;
            else
                tapeLatch = false;
        }
        controls.tapeStop = !altMode && (tapeLatch || fsCtl[1].Pressed());

        // ---- knob 1 addresses volume, or dry amount on the alt layer ----
        if(altMode)
        {
            // Move to engage, per knob. Each one is inert until turned past
            // kAltDeadband from where it sat on entry - a deliberate nudge,
            // well clear of ADC jitter or knocking the pedal.
            for(int i = 0; i < 6; i++)
                if(!altArmed[i] && fabsf(controls.knob[i] - altEntry[i]) > kAltDeadband)
                    altArmed[i] = true;

            // knob 1 - how much dry the dry switch lets through
            if(altArmed[0])
                controls.dryAmount = controls.knob[0];

            // knob 2 - lag: how much the modulation glides between values
            if(altArmed[1])
                controls.lag = controls.knob[1];

            // knobs 4/5/6 - Condor-style three band EQ
            if(altArmed[3])
                controls.eqLow = controls.knob[3];
            if(altArmed[4])
                controls.eqMid = controls.knob[4];
            if(altArmed[5])
                controls.eqHigh = controls.knob[5];

            // knob 3 - modulation waveshape
            if(altArmed[2])
            {
                int sh = (int)(controls.knob[2] * (float)LFO_COUNT);
                if(sh >= LFO_COUNT)
                    sh = LFO_COUNT - 1;
                controls.lfoShape = sh;
            }
        }
        else
        {
            for(int i = 0; i < 6; i++)
                if(!normArmed[i] && fabsf(controls.knob[i] - normEntry[i]) > kAltDeadband)
                    normArmed[i] = true;

            if(normArmed[0])
                controls.volume = controls.knob[0];
            if(normArmed[1])
                controls.wowDepth = controls.knob[1];
            if(normArmed[2])
                controls.rate = controls.knob[2];
            if(normArmed[3])
                controls.sat = controls.knob[3];
            if(normArmed[4])
                controls.flavour = controls.knob[4];
            if(normArmed[5])
                controls.fail = controls.knob[5];
        }

        // ---- LEDs -------------------------------------------------------
        if(now < flashUntil)
        {
            // Three flashes on both LEDs to confirm the reset landed.
            const bool on = ((now - flashStart) % 300) < 150;
            ledOut[0].Write(on);
            ledOut[1].Write(on);
        }
        else if(altMode)
        {
            // Both blink together while the alt layer is live, and stay solid
            // once the knob has been caught and is actually moving the value.
            // Both LEDs blink for the whole time the alt layer is held.
            const bool fast = (now % 300) < 150;
            ledOut[0].Write(fast);
            ledOut[1].Write(fast);
        }
        else
        {
            ledOut[0].Write(controls.effectOn);
            ledOut[1].Write(controls.tapeStop);
        }

        System::Delay(1);
    }
}
