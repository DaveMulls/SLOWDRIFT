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
//   fs1    effect on/off (latching)
//   fs2    tape stop (hold)

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
    SD_TAPE_LEN    = 6200,  //  128 ms
    SD_FLANGE_LEN  = 800,   //   15 ms
    SD_SLAP_LEN    = 7400,  //  150 ms
    SD_TSTAPE_LEN  = 29000, //  600 ms - the big one, put this in SDRAM
    SD_TSPITCH_LEN = 6200   //  128 ms
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
    bool  dryOn    = false; // sw1
    bool  compOn   = false; // sw2
    bool  bloomOn  = false; // sw3
    bool  vibrato  = false; // sw4
    bool  effectOn = false; // fs1, latching
    bool  tapeStop = false; // fs2, held
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

    // --- bloom (lpg_engine)
    PdHip   lpgHp;
    PdLop   lpgAtk, lpgRel, lpgRipple, lpgKnob;
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
    PdPhasor      vibLfo;
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
    int   currentMode = -1;
    // mode 1: harmonic
    PdLop   f1a, f1b;
    PdBp    f1bp;
    // mode 2: alt EQ
    PdHip   f2h1, f2h2;
    PdLop   f2l1, f2l2;
    PdBp    f2bp;
    // mode 3: tape flange
    PdDelay flange;
    PdLop         flangeMod1, flangeMod2;
    // mode 4: tremolo
    PdPhasor tremLfo;
    PdLop    tremDepth;
    // mode 5: slap
    PdDelay slap;
    PdLop         slapLp;

    // --- tape stop
    PdDelay tsTape;
    PdDelay tsPitch;
    PdLine         tsRate, tsTapeGain;
    float          tsDelayMs = 2.f;
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
              float* bSlap,
              float* bTsTape,
              float* bTsPitch)
    {
        sr = sampleRate;

        inHp.Init(sr, 10.f);
        outHp.Init(sr, 10.f);
        preHp50.Init(sr, 50.f);
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
        walkS1.Init(sr, 0.5f);
        walkS2.Init(sr, 0.5f);
        walkHp.Init(sr, 0.3f);
        sine1.Init(sr);
        sine1.SetFreq(0.61f);
        sine2.Init(sr);
        sine2.SetFreq(1.07f);
        sine2.SetPhase(0.37f);
        sine3.Init(sr);
        sine3.SetFreq(1.83f);
        sine3.SetPhase(0.71f);
        vibLfo.Init(sr);
        vibDepth.Init(sr, 10.f);
        altDepth.Init(sr, 20.f);
        snagSm1.Init(sr, 1.f);
        snagSm2.Init(sr, 1.f);
        dropoutSm.Init(sr, 10.f);

        for(int i = 0; i < 6; i++)
            modeGain[i].Init(sr, i == 0 ? 1.f : 0.f);

        f1a.Init(sr, 1000.f);
        f1b.Init(sr, 500.f);
        f1bp.Init(sr, 600.f, 10.f);
        f2h1.Init(sr, 500.f);
        f2h2.Init(sr, 500.f);
        f2l1.Init(sr, 1800.f);
        f2l2.Init(sr, 1800.f);
        f2bp.Init(sr, 1000.f, 0.2f);
        flange.Init(sr, bFlange, SD_FLANGE_LEN);
        flangeMod1.Init(sr, 0.3f);
        flangeMod2.Init(sr, 0.3f);
        tremLfo.Init(sr);
        tremDepth.Init(sr, 10.f);
        slap.Init(sr, bSlap, SD_SLAP_LEN);
        slapLp.Init(sr, 4000.f);

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
        int mode = (int)(c.knob[4] * 6.f);
        if(mode > 5)
            mode = 5;
        if(mode != currentMode)
        {
            currentMode = mode;
            for(int i = 0; i < 6; i++)
                modeGain[i].Set(i == mode ? 1.f : 0.f, 0.5f);
        }

        // LFO rates follow knob3
        vibLfo.SetFreq(0.15f + c.knob[2] * 6.8f);
        tremLfo.SetFreq(0.5f + c.knob[2] * 14.5f);

        // knob6 drives either bloom or failure depending on the bloom switch.
        // The inactive one holds its last position rather than following along.
        if(c.bloomOn)
            bloomAmt = c.knob[5];
        else
            failAmt = c.knob[5];

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
                tsRate.Set(1.f, 350.f);     // accelerate back up: pitch rises
                tsTapeGain.Set(1.f, 120.f);
                tsStage = 1;
                tsTimer = (int)(0.350f * sr);
            }
        }
        if(tsStage == 1 && (tsTimer -= blockSize) <= 0)
        {
            // Spinning up leaves an accumulated lag. Bleed it off slightly
            // fast so the delay walks back to minimum as a slow tape drift.
            tsRate.Set(1.06f, 20.f);
            tsStage = 2;
        }
        if(tsStage == 2 && tsDelayMs <= 2.5f)
        {
            tsRate.Set(1.f, 20.f);
            tsStage = -1;
        }
    }

    inline float Process(float in, const Controls& c)
    {
        const float dry = inHp.Process(in);

        if(!c.effectOn)
            return Clampf(dry, -0.99f, 0.99f);

        // ---------------- bloom / LPG (vactrol model) --------------------
        const float lpgIn  = lpgHp.Process(dry);
        const float rect   = fabsf(lpgIn);
        const float atk    = lpgAtk.Process(rect);
        const float rel    = lpgRel.Process(rect);
        const float env    = lpgRipple.Process(atk > rel ? atk : rel);
        const float knob   = lpgKnob.Process(bloomAmt);
        // Sensitivity raised so transients swing the gate, and the resting
        // floor reduced so it still breathes with the knob near maximum.
        const float open_  = Clampf(Clampf(env * 14.f, 0.f, 1.f) * knob
                                        + knob * knob * 0.4f,
                                    0.f, 1.f);
        const float o2     = open_ * open_;
        const float cutoff = o2 * 0.98f + 0.02f; // fully open at the top now
        const float fb     = 1.f - cutoff;
        float       lpg    = lpgP1.Process(lpgIn * cutoff, fb);
        lpg                = lpgP2.Process(lpg * cutoff, fb);
        lpg *= (open_ * 0.65f + 0.35f);

        // ---------------- pre-saturation sum -----------------------------
        // bloom is summed with the dry path, then high-passed at 50 Hz
        const float x = preHp50.Process(lpg + dry);

        // ---------------- drive + saturation -----------------------------
        const float drive = driveSm.Process(1.f + c.knob[3] * 24.f);
        float pre = x * drive * 0.5f;
        pre += 0.06f * pre * pre;                 // slight asymmetry, 2nd harmonic
        // Cubic soft clip: essentially linear at low level, so the knob is
        // genuinely clean at zero, then thickens progressively.
        float sat = (pre > 1.f) ? 1.f
                                : ((pre < -1.f) ? -1.f : 1.5f * (pre - pre * pre * pre / 3.f));
        sat /= 0.75f * sqrtf(drive);              // level compensation
        // Tape loses top end as it saturates: 12 kHz down to about 3.5 kHz.
        satTilt.SetFreq(12000.f - c.knob[3] * 8500.f);
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
        const float ped = c.knob[2] * 6.f;
        n               = Clampf(n, -ped, 18.f);
        const float k3   = c.knob[2] / 3.f;
        const float wowN = (n + c.knob[2] * 6.f) * k3;

        // random-walk branch
        if(--walkTimer <= 0)
        {
            walkTimer = (int)((0.500f + Rand01()) * sr);
            walkTarget = Rand01() * 2.f - 1.f;
            walkLine.Set(walkTarget, 800.f + c.knob[1] * 1200.f);
        }
        float w = walkS2.Process(walkS1.Process(walkLine.Process()));
        w       = (w - walkHp.Process(w)) * 1.05f;

        // sine trio
        const float s = (cosf(6.2831853f * sine1.Process())
                         + cosf(6.2831853f * sine2.Process())
                         + cosf(6.2831853f * sine3.Process()))
                        * 0.221f;

        const float depth = altDepth.Process(c.knob[1]);
        const float mod   = Clampf(w + s, -1.f, 1.f);
        const float wowA  = mod * depth * 5.f + depth * 5.f;

        // vibrato branch (replaces wow when sw4 is up)
        const float vdep = vibDepth.Process(c.knob[1] * 3.f);
        const float vib  = cosf(6.2831853f * vibLfo.Process()) * vdep;

        const float snag = snagSm2.Process(snagSm1.Process(snagTime));

        float delayMs;
        if(c.vibrato)
            delayMs = 4.f + vib + snag;
        else
            delayMs = wowN + wowA + snag;
        delayMs = Clampf(delayMs, 1.f, 120.f);

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

        // 1 - harmonic saturator
        {
            const float d  = FastTanh(f * 1.5f);
            const float lo = f1b.Process(f1a.Process(d)) * 2.f;
            const float hi = FastTanh(f1bp.Process(d) * 10.f);
            bus += (lo + hi) * 0.2f * g1;
        }

        // 2 - alt EQ
        {
            const float band = f2l2.Process(f2l1.Process(f2h2.Process(f2h1.Process(f))));
            const float peak = f2bp.Process(band) * 6.f;
            bus += (band + peak) * 0.35f * g2; // was 3.5 dB hotter than the rest
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
            const float td = tremDepth.Process(c.knob[1] * 0.5f);
            const float lf = cosf(6.2831853f * tremLfo.Process()) * td + (1.f - td);
            bus += f * lf * g4;
        }

        // 5 - slap
        {
            slap.Write(f);
            const float sv = slapLp.Process(slap.ReadMs(120.f)) * 0.5f;
            bus += (f + sv) * g5;
        }

        // ---------------- dry blend --------------------------------------
        // Pd sums this into the bus ahead of the tape stop, so it is subject
        // to the tape-stop envelope and the output level knob like everything
        // else. Adding it after the level control made it far too loud.
        if(c.dryOn)
            bus += comp * 0.25f; // roughly level with the wet bus

        // ---------------- tape stop --------------------------------------
        tsTape.Write(bus);
        tsPitch.Write(bus);

        const float rate = tsRate.Process();
        tsDelayMs += (1.f - rate) * 1000.f / sr;
        tsDelayMs = Clampf(tsDelayMs, 2.f, 560.f);
        float stopped = tsTape.ReadMs(tsDelayMs) * tsTapeGain.Process();

        // ---------------- output -----------------------------------------
        float outv = stopped * (c.knob[0] * 2.f); // noon = unity
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
static float               bufSlap[SD_SLAP_LEN];
static float               bufTsPitch[SD_TSPITCH_LEN];

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
    pedal.Init(sr, bufTape, bufFlange, bufSlap, bufTsTape, bufTsPitch);

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
        if(fsCtl[0].RisingEdge())
            controls.effectOn = !controls.effectOn;
        controls.tapeStop = fsCtl[1].Pressed();

        ledOut[0].Write(controls.effectOn);
        ledOut[1].Write(controls.tapeStop);

        System::Delay(1);
    }
}
