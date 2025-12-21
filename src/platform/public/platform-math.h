#ifndef DMT_PLATFORM_PUBLIC_PLATFORM_MATH_H
#define DMT_PLATFORM_PUBLIC_PLATFORM_MATH_H
#include <stdlib.h>
#include <cassert>
#include <random>
#include <iostream>
#include <cmath>

#define N 16
#define MASK ((unsigned)(1 << (N - 1)) + (1 << (N - 1)) - 1)
#define LOW(x) ((unsigned)(x) & MASK)
#define HIGH(x) LOW((x) >> N)
#define MUL(x, y, z)                \
  {                                 \
    long l = (long)(x) * (long)(y); \
    (z)[0] = LOW(l);                \
    (z)[1] = HIGH(l);               \
  }
#define CARRY(x, y) ((long)(x) + (long)(y) > MASK)
#define ADDEQU(x, y, z) (z = CARRY(x, (y)), x = LOW(x + (y)))
#define X0 0x330E
#define X1 0xABCD
#define X2 0x1234
#define A0 0xE66D
#define A1 0xDEEC
#define A2 0x5
#define C 0xB
#define SET3(x, x0, x1, x2) ((x)[0] = (x0), (x)[1] = (x1), (x)[2] = (x2))
#define SETLOW(x, y, n) \
  SET3(x, LOW((y)[n]), LOW((y)[(n) + 1]), LOW((y)[(n) + 2]))
#define SEED(x0, x1, x2) (SET3(x, x0, x1, x2), SET3(a, A0, A1, A2), c = C)

namespace dmt::math {
static unsigned x[3] = {X0, X1, X2}, a[3] = {A0, A1, A2}, c = C;
static unsigned short lastx[3];
static void next() {
  unsigned p[2], q[2], r[2], carry0, carry1;

  MUL(a[0], x[0], p);
  ADDEQU(p[0], c, carry0);
  ADDEQU(p[1], carry0, carry1);
  MUL(a[0], x[1], q);
  ADDEQU(p[1], q[0], carry0);
  MUL(a[1], x[0], r);
  x[2] = LOW(carry0 + carry1 + CARRY(p[1], r[0]) + q[1] + r[1] + a[0] * x[2] +
             a[1] * x[1] + a[2] * x[0]);
  x[1] = LOW(p[1] + r[0]);
  x[0] = LOW(p[0]);
}

void srand48(long seedval) { SEED(X0, LOW(seedval), HIGH(seedval)); }

unsigned short* seed48(unsigned short seed16v[3]) {
  SETLOW(lastx, x, 0);
  SEED(LOW(seed16v[0]), LOW(seed16v[1]), LOW(seed16v[2]));
  return (lastx);
}

long lrand48() {
  next();
  return (((long)x[2] << (N - 1)) + (x[1] >> 1));
}

// implement drand48()
// use the linear congruential formula: x(n+1)= (a*x(n) + c)%m
// where a = 0x5DEECE66D, c = 0xB m = 2^48 = 1<<48
// create an internal buffer to store the last 48bit of x(i) previously
// generated The initialiser function srand48() sets the high-order 32 bits of
// Xi to the low-order 32 bits contained in its argument.
//  The low-order 16 bits of Xi are set to the arbitrary value 330E16 .

double drand48() {
  static double two16m = 1.0 / (1L << N);

  next();
  return (two16m * (two16m * (two16m * x[0] + x[1]) + x[2]));
}

double RandN() {
  static bool cached = false;
  static double cn;

  if (cached) {
    cached = false;
    return cn;
  }
  // note: The functions drand48() return nonnegative, double-precision,
  // floating-point values, uniformly distributed over the interval [0.0,1.0).
  double a = std::sqrt(-2 * std::log(drand48()));
  double b = 6.283185307179586476925286766559 * drand48();
  cn = sin(b) * a;
  cached = true;
  return cos(b) * a;
}

class RNG {
 protected:
  // pointers to the start and end of outputs per transform vector
  float *m_rngOutputCurr, *m_rngOutputEnd;

  // must leave at least one value in curr..end that can be returned
  virtual void UpdateRngOutput() = 0;

 public:
  RNG() : m_rngOutputCurr(0), m_rngOutputEnd(0) {}

  virtual ~RNG() {}

  virtual char const* Name() = 0;
  virtual char const* Description() = 0;

  virtual void Generate(unsigned count, float* values) = 0;

  // these have been pulled back from being virtual functions for
  // performance reasons. Now they can be inlined into the simulations.
  float Generate() {
    if (m_rngOutputCurr >= m_rngOutputEnd) {
      UpdateRngOutput();
      assert(m_rngOutputCurr < m_rngOutputEnd);
    }
    return *m_rngOutputCurr++;
  }
  float operator()() { return Generate(); }
};

template <unsigned tBATCH_SIZE>
class BatchedRngBase : public RNG {
 public:
  enum { BATCH_SIZE = tBATCH_SIZE };

 private:
  float m_output[BATCH_SIZE];
  float* m_outputReadPos;

 protected:
  virtual void GenerateImpl(float* dest) = 0;

  unsigned OutputSamplesLeft() {
    assert(m_outputReadPos >= m_output &&
           m_outputReadPos <= m_output + BATCH_SIZE);
    return (m_output + BATCH_SIZE) - m_outputReadPos;
  }
  virtual void UpdateRngOutput() {
    GenerateImpl(m_output);
    m_rngOutputCurr = m_output;
  }

 public:
  BatchedRngBase() : m_outputReadPos(m_output + BATCH_SIZE) {}

  void Generate(unsigned count, float* values) {
    if (count > OutputSamplesLeft()) {
      std::copy(m_outputReadPos, m_output + BATCH_SIZE, values);
      count -= OutputSamplesLeft();
      values += OutputSamplesLeft();
      m_outputReadPos = m_output + BATCH_SIZE;  // -> OutputSamplesLeft()==0

      // transform directly into output buffer
      while (count >= BATCH_SIZE) {
        GenerateImpl(values);
        count -= BATCH_SIZE;
        values += BATCH_SIZE;
      }

      // always leave scope with a full pool
      GenerateImpl(m_output);
      m_outputReadPos = m_output;
    }

    assert(count <= OutputSamplesLeft());

    std::copy(m_outputReadPos, m_outputReadPos + count, values);
    m_outputReadPos += count;
  }

  float Generate() {
    if (OutputSamplesLeft() == 0) {
      GenerateImpl(m_output);
      m_outputReadPos = m_output;
    }
    return *m_outputReadPos++;
  }

  float operator()() { return Generate(); }
};

typedef void (*cuda_nrng_func_t)(
    unsigned STATE_SIZE,  // size of each thread's state size
    unsigned RNG_COUNT,  // number of rngs (i.e. total threads across all grids)
    unsigned PER_RNG_OUTPUT_COUNT,  // number of outputs for each RNG
    unsigned* state,  // [in,out] STATE_SIZE*RNG_COUNT  On output is assumed to
                      // contain updated state.
    float* output     // [out] RNG_COUNT*PER_RNG_OUTPUT_SIZE
);

template <cuda_nrng_func_t RNG_FUNC, unsigned STATE_SIZE, unsigned RNG_COUNT,
          unsigned PER_RNG_OUTPUT_COUNT>
class NrngCUDA : public BatchedRngBase<RNG_COUNT * PER_RNG_OUTPUT_COUNT> {
 private:
  // Mersenne Twister pseudorandom number generator
  std::mt19937 m_mt;

  unsigned m_states[STATE_SIZE * RNG_COUNT];

  char const *m_name, *m_desc;

  void InitSeeds() {
    for (unsigned i = 0; i < STATE_SIZE * RNG_COUNT; i++) {
      do {
        m_states[i] = m_mt();
      } while (m_states[i] < 128);
    }
  }

 protected:
  virtual void GenerateImpl(float* dest) {
    RNG_FUNC(STATE_SIZE, RNG_COUNT, PER_RNG_OUTPUT_COUNT, m_states, dest);
  }

 public:
  NrngCUDA(char const* name, char const* desc)
      : m_mt(lrand48()), m_name(name), m_desc(desc) {
    std::random_device rd;
    m_mt.seed(rd());
    InitSeeds();
  }

  virtual char const* Name() { return m_name; }
  virtual char const* Description() { return m_desc; }
};

};  // namespace dmt::math
#endif  // DMT_PLATFORM_PUBLIC_PLATFORM_MATH_H
