#![no_std]

extern crate alloc;

use core::{
    fmt::{self, Debug},
    ops::AddAssign,
};

use num_traits::{cast::FromPrimitive, float::Float, identities::One, identities::Zero};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A statistics object that continuously calculates min, max, mean, and deviation for tracking time-varying statistics.
/// Utilizes Welford's Online algorithm. More details on the algorithm can be found at:
/// "https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
///
///
/// # Example
///
/// ```
/// use rolling_stats::Stats;
/// use rand_distr::{Distribution, Normal};
/// use rand::SeedableRng;
///
/// type T = f64;
///
/// const MEAN: T = 0.0;
/// const STD_DEV: T = 1.0;
/// const NUM_SAMPLES: usize = 10_000;
/// const SEED: u64 = 42;
///
/// let mut stats: Stats<T> = Stats::new();
/// let mut rng = rand::rngs::StdRng::seed_from_u64(SEED); // Seed the RNG for reproducibility
/// let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();
///
/// // Generate random data
/// let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();
///
/// // Update the stats one by one
/// random_data.iter().for_each(|v| stats.update(*v));
///
/// // Print the stats
/// println!("{}", stats);
/// // Output: (avg: 0.00, std_dev: 1.00, min: -3.53, max: 4.11, count: 10000)
///
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Stats<T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug> {
    /// The smallest value seen so far.
    pub min: T,

    /// The largest value seen so far.
    pub max: T,

    /// The calculated mean (average) of all the values seen so far.
    pub mean: T,

    /// The calculated standard deviation of all the values seen so far.
    pub std_dev: T,

    /// The count of the total values seen.
    pub count: usize,

    /// The square of the mean value. This is an internal value used in the calculation of the standard deviation.
    mean2: T,
}

/// Implementing the Display trait for the Stats struct to present the statistics in a readable format.
impl<T> fmt::Display for Stats<T>
where
    T: fmt::Display + Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    /// Formats the output of the statistics.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(2);

        write!(f, "(avg: {:.precision$}, std_dev: {:.precision$}, min: {:.precision$}, max: {:.precision$}, count: {})", self.mean, self.std_dev, self.min, self.max, self.count, precision=precision)
    }
}

impl<T> Default for Stats<T>
where
    T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    fn default() -> Stats<T> {
        Stats::new()
    }
}

impl<T> Stats<T>
where
    T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    /// Creates a new stats object with all values set to their initial states.
    pub fn new() -> Stats<T> {
        Stats {
            count: 0,
            min: T::infinity(),
            max: T::neg_infinity(),
            mean: T::zero(),
            std_dev: T::zero(),
            mean2: T::zero(),
        }
    }

    /// Updates the stats object with a new value. The statistics are recalculated using the new value.
    pub fn update(&mut self, value: T) {
        // Track min and max
        if value > self.max {
            self.max = value;
        }
        if value < self.min {
            self.min = value;
        }

        // Increment counter
        self.count += 1;
        let count = T::from(self.count).unwrap();

        // Calculate mean
        let delta = value - self.mean;
        self.mean += delta / count;

        // Mean2 used internally for standard deviation calculation
        let delta2 = value - self.mean;
        self.mean2 += delta * delta2;

        // Calculate standard deviation
        if self.count > 1 {
            self.std_dev = (self.mean2 / (count - T::one())).sqrt();
        }
    }

    /// Merges another stats object into new one. This is done by combining the statistics of the two objects
    /// in accordance with the formula provided at:
    /// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    ///
    /// This is useful for combining statistics from multiple threads or processes.
    ///
    /// # Example
    ///
    /// ```
    /// use rolling_stats::Stats;
    /// use rand_distr::{Distribution, Normal};
    /// use rand::SeedableRng;
    /// use rayon::prelude::*;
    ///
    /// type T = f64;
    ///
    /// const MEAN: T = 0.0;
    /// const STD_DEV: T = 1.0;
    /// const NUM_SAMPLES: usize = 500_000;
    /// const SEED: u64 = 42;
    /// const CHUNK_SIZE: usize = 1000;
    ///
    /// let mut stats: Stats<T> = Stats::new();
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(SEED); // Seed the RNG for reproducibility
    /// let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();
    ///
    /// // Generate random data
    /// let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();
    ///
    /// // Update the stats in parallel. New stats objects are created for each chunk of data.
    /// let stats: Vec<Stats<T>> = random_data
    ///     .par_chunks(CHUNK_SIZE) // Multi-threaded parallelization via Rayon
    ///     .map(|chunk| {
    ///             let mut s: Stats<T> = Stats::new();
    ///             chunk.iter().for_each(|v| s.update(*v));
    ///             s
    ///      })
    ///     .collect();
    ///
    /// // Check if there's more than one stat object
    /// assert!(stats.len() > 1);
    ///
    /// // Accumulate the stats using the reduce method. The last stats object is returned.
    /// let merged_stats = stats.into_iter().reduce(|acc, s| acc.merge(&s)).unwrap();
    ///
    /// // Print the stats
    /// println!("{}", merged_stats);
    ///
    /// // Output: (avg: -0.00, std_dev: 1.00, min: -4.53, max: 4.57, count: 500000)
    ///```
    pub fn merge(&self, other: &Self) -> Self {
        let mut merged = Stats::<T>::new();

        // If both stats objects are empty, return an empty stats object
        if self.count + other.count == 0 {
            return merged;
        }

        // If one of the stats objects is empty, return the other one
        if self.count == 0 {
            return other.clone();
        } else if other.count == 0 {
            return self.clone();
        }

        merged.max = if other.max > self.max {
            other.max
        } else {
            self.max
        };

        merged.min = if other.min < self.min {
            other.min
        } else {
            self.min
        };

        merged.count = self.count + other.count;

        // Convert to T to avoid overflow
        let merged_count = T::from(merged.count).unwrap();
        let self_count = T::from(self.count).unwrap();
        let other_count = T::from(other.count).unwrap();

        let delta = other.mean - self.mean;

        merged.mean = (self.mean * self_count + other.mean * other_count) / merged_count;

        merged.mean2 =
            self.mean2 + other.mean2 + delta * delta * self_count * other_count / merged_count;

        merged.std_dev = (merged.mean2 / (merged_count - T::one())).sqrt();

        merged
    }
}

/// Extended stats with exponential decay and adaptive alpha
/// Builds on top of the standard Welford algorithm
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct AdaptiveStats<T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug> {
    /// Core Welford stats (unchanged)
    core: Stats<T>,
    
    /// Exponential decay parameters
    base_alpha: T,           // Base alpha (e.g., 0.02)
    current_alpha: T,        // Current adaptive alpha
    decay_factor: T,         // How much to decay old samples (e.g., 0.999)
    
    /// Adaptive alpha parameters  
    alpha_min: T,            // Minimum alpha (e.g., 0.001)
    alpha_max: T,            // Maximum alpha (e.g., 0.10)
    volatility_sensitivity: T, // How much volatility affects alpha (e.g., 2.0)
    
    /// Enhanced outputs
    stable_variance: T,      // Long-term stable variance baseline
    recent_variance: T,      // Short-term recent variance
    trend_factor: T,         // Recent/Stable ratio (1.0 = normal, >1 = trending)
    stability_score: T,      // How stable recent samples are (0-1)
    
    /// Internal state
    samples_since_reset: usize,
    warmup_samples: usize,   // Samples needed before adaptive behavior kicks in
}

impl<T> Default for AdaptiveStats<T>
where
    T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AdaptiveStats<T>
where
    T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    /// Create new adaptive stats with default parameters
    pub fn new() -> Self {
        Self::with_params(
            T::from(0.02).unwrap(),   // base_alpha
            T::from(0.999).unwrap(),  // decay_factor  
            T::from(0.001).unwrap(),  // alpha_min
            T::from(0.10).unwrap(),   // alpha_max
            T::from(2.0).unwrap(),    // volatility_sensitivity
            100                       // warmup_samples
        )
    }
    
    /// Create with custom parameters
    pub fn with_params(
        base_alpha: T,
        decay_factor: T, 
        alpha_min: T,
        alpha_max: T,
        volatility_sensitivity: T,
        warmup_samples: usize
    ) -> Self {
        Self {
            core: Stats::new(),
            base_alpha,
            current_alpha: base_alpha,
            decay_factor,
            alpha_min,
            alpha_max,
            volatility_sensitivity,
            stable_variance: T::zero(),
            recent_variance: T::zero(),
            trend_factor: T::one(),
            stability_score: T::one(),
            samples_since_reset: 0,
            warmup_samples,
        }
    }
    
    /// Standard update (uses base alpha)
    pub fn update(&mut self, value: T) {
        self.update_with_alpha(value, self.base_alpha)
    }
    
    /// Update with custom alpha multiplier
    /// multiplier: 1.0 = normal, >1.0 = more reactive, <1.0 = more stable
    pub fn update_with_multiplier(&mut self, value: T, multiplier: T) {
        let alpha = (self.base_alpha * multiplier)
            .max(self.alpha_min)
            .min(self.alpha_max);
        self.update_with_alpha(value, alpha)
    }
    
    /// Core update with explicit alpha
    pub fn update_with_alpha(&mut self, value: T, alpha: T) {
        // Update core stats with decay
        self.update_core_with_decay(value, alpha);
        
        // Update enhanced metrics
        self.update_enhanced_metrics(value);
        
        // Adapt alpha based on recent volatility
        self.adapt_alpha();
        
        self.samples_since_reset += 1;
    }
    
    /// Update with automatic alpha adaptation
    /// The alpha adapts based on recent volatility vs stable baseline
    pub fn update_adaptive(&mut self, value: T) {
        self.update_with_alpha(value, self.current_alpha)
    }
}

impl<T> AdaptiveStats<T>
where
    T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    fn update_core_with_decay(&mut self, value: T, alpha: T) {
        if self.core.count == 0 {
            // First sample - initialize normally
            self.core.update(value);
            self.stable_variance = T::zero();
            self.recent_variance = T::zero();
            return;
        }
        
        // Apply exponential decay to existing stats
        let decay = T::one() - alpha;
        
        // Decay the count (conceptually)
        let effective_count = T::from(self.core.count).unwrap() * decay + T::one();
        
        // Update min/max normally
        if value > self.core.max { self.core.max = value; }
        if value < self.core.min { self.core.min = value; }
        
        // Exponentially decayed mean update
        let delta = value - self.core.mean;
        self.core.mean += alpha * delta;
        
        // Exponentially decayed variance update  
        let delta2 = value - self.core.mean;
        self.core.mean2 = self.core.mean2 * decay + alpha * delta * delta2;
        
        // Update standard deviation
        if effective_count > T::one() {
            self.core.std_dev = (self.core.mean2 / (effective_count - T::one())).sqrt();
        }
        
        // Increment count (for compatibility)
        self.core.count += 1;
    }
    
    fn update_enhanced_metrics(&mut self, value: T) {
        let alpha_stable = T::from(0.005).unwrap(); // Very slow for stable baseline
        let alpha_recent = T::from(0.05).unwrap();  // Faster for recent
        
        let variance = if self.core.count > 1 {
            let delta = value - self.core.mean;
            delta * delta
        } else {
            T::zero()
        };
        
        // Update stable baseline (very slow adaptation)
        if self.samples_since_reset == 1 {
            self.stable_variance = variance;
        } else {
            self.stable_variance = self.stable_variance * (T::one() - alpha_stable) + variance * alpha_stable;
        }
        
        // Update recent variance (faster adaptation)
        if self.samples_since_reset == 1 {
            self.recent_variance = variance;
        } else {
            self.recent_variance = self.recent_variance * (T::one() - alpha_recent) + variance * alpha_recent;
        }
        
        // Calculate trend factor
        if self.stable_variance > T::zero() {
            self.trend_factor = self.recent_variance / self.stable_variance;
        }
        
        // Calculate stability score (inverse of trend factor, clamped)
        self.stability_score = T::one() / (T::one() + (self.trend_factor - T::one()).abs());
    }
    
    fn adapt_alpha(&mut self) {
        if self.samples_since_reset < self.warmup_samples {
            return; // Don't adapt during warmup
        }
        
        // Higher volatility = higher alpha (more reactive)
        // Lower volatility = lower alpha (more stable)
        let volatility_ratio = if self.trend_factor > T::one() {
            // Trending: increase alpha
            (self.trend_factor - T::one()) * self.volatility_sensitivity + T::one()
        } else {
            // Stable: keep alpha low  
            T::one() / ((T::one() - self.trend_factor) * self.volatility_sensitivity + T::one())
        };
        
        self.current_alpha = (self.base_alpha * volatility_ratio)
            .max(self.alpha_min)
            .min(self.alpha_max);
    }
}

impl<T> AdaptiveStats<T>
where
    T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    /// Get the core stats (for compatibility)
    pub fn core(&self) -> &Stats<T> { &self.core }
    
    /// Standard getters (delegate to core)
    pub fn count(&self) -> usize { self.core.count }
    pub fn mean(&self) -> T { self.core.mean }
    pub fn std_dev(&self) -> T { self.core.std_dev }
    pub fn min(&self) -> T { self.core.min }
    pub fn max(&self) -> T { self.core.max }
    pub fn variance(&self) -> T { self.core.std_dev * self.core.std_dev }
    
    /// Enhanced getters
    pub fn stable_std(&self) -> T { self.stable_variance.sqrt() }
    pub fn stable_variance(&self) -> T { self.stable_variance }
    pub fn recent_std(&self) -> T { self.recent_variance.sqrt() }
    pub fn recent_variance(&self) -> T { self.recent_variance }
    pub fn trend_factor(&self) -> T { self.trend_factor }
    pub fn stability_score(&self) -> T { self.stability_score }
    pub fn current_alpha(&self) -> T { self.current_alpha }
    
    /// Convenience methods
    pub fn is_trending(&self) -> bool {
        self.trend_factor > T::from(1.2).unwrap()
    }
    
    pub fn is_stable(&self) -> bool {
        self.stability_score > T::from(0.8).unwrap()
    }
    
    /// Calculate Z-score using stable baseline
    pub fn z_score_stable(&self, value: T) -> T {
        if self.stable_variance > T::zero() {
            (value - self.core.mean) / self.stable_std()
        } else {
            T::zero()
        }
    }
    
    /// Calculate percentile approximation (assumes normal distribution)
    pub fn percentile_approx(&self, p: T) -> T {
        // Simple approximation: mean + z_score * std
        // For p=0.9 (90th percentile), z ≈ 1.28
        let z_score = match p {
            p if p >= T::from(0.95).unwrap() => T::from(1.96).unwrap(),  // 95%
            p if p >= T::from(0.90).unwrap() => T::from(1.28).unwrap(),  // 90%
            p if p >= T::from(0.75).unwrap() => T::from(0.67).unwrap(),  // 75%
            _ => T::from(0.5).unwrap()  // ~60%
        };
        
        self.core.mean + z_score * self.stable_std()
    }
    
    /// Reset all state (keep parameters)
    pub fn reset(&mut self) {
        self.core = Stats::new();
        self.current_alpha = self.base_alpha;
        self.stable_variance = T::zero();
        self.recent_variance = T::zero();
        self.trend_factor = T::one();
        self.stability_score = T::one();
        self.samples_since_reset = 0;
    }
}

impl<T> fmt::Display for AdaptiveStats<T>
where
    T: fmt::Display + Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);
        
        write!(f, 
            "(avg: {:.precision$}, stable_std: {:.precision$}, recent_std: {:.precision$}, trend: {:.precision$}, stability: {:.precision$}, count: {}, alpha: {:.precision$})",
            self.mean(), 
            self.stable_std(), 
            self.recent_std(),
            self.trend_factor(),
            self.stability_score(),
            self.count(),
            self.current_alpha(),
            precision = precision
        )
    }
}

impl<T> Stats<T>
where
    T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    /// Creates a new stats object with a prior value (as if we had one sample)
    pub fn with_prior(value: T) -> Stats<T> {
        let mut stats = Stats::new();
        stats.update(value);
        stats
    }

    /// Sets a prior value, resetting all statistics as if this was the first sample
    pub fn set_prior(&mut self, value: T) {
        *self = Stats::new();
        self.update(value);
    }

    /// Sets a prior with custom count (as if we had 'count' samples of this value)
    pub fn set_prior_with_count(&mut self, value: T, count: usize) {
        *self = Stats::new();
        for _ in 0..count {
            self.update(value);
        }
    }

    /// Resets stats but keeps the same initial value as prior
    pub fn reset_with_prior(&mut self, prior_value: T) {
        let old_prior = self.mean; // Use current mean as reference
        self.set_prior(prior_value);
    }
}

impl<T> AdaptiveStats<T>
where
    T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    /// Creates new adaptive stats with a prior value
    pub fn with_prior(prior_value: T) -> Self {
        let mut stats = Self::new();
        stats.set_prior(prior_value);
        stats
    }

    /// Creates new adaptive stats with custom parameters and prior
    pub fn with_params_and_prior(
        prior_value: T,
        base_alpha: T,
        decay_factor: T,
        alpha_min: T,
        alpha_max: T,
        volatility_sensitivity: T,
        warmup_samples: usize
    ) -> Self {
        let mut stats = Self::with_params(
            base_alpha, 
            decay_factor, 
            alpha_min, 
            alpha_max, 
            volatility_sensitivity, 
            warmup_samples
        );
        stats.set_prior(prior_value);
        stats
    }

    /// Sets a prior value, initializing all statistics
    pub fn set_prior(&mut self, value: T) {
        // Reset core stats with prior
        self.core = Stats::with_prior(value);
        
        // Initialize enhanced metrics with the prior
        let variance = T::zero(); // No variance with single sample
        self.stable_variance = variance;
        self.recent_variance = variance;
        self.trend_factor = T::one();
        self.stability_score = T::one();
        self.current_alpha = self.base_alpha;
        self.samples_since_reset = 1; // We have one sample (the prior)
    }

    /// Sets prior with estimated variance (useful for initialization)
    pub fn set_prior_with_variance(&mut self, mean_value: T, variance_estimate: T) {
        // Set the mean via prior
        self.set_prior(mean_value);
        
        // Override the variance estimates
        self.stable_variance = variance_estimate;
        self.recent_variance = variance_estimate;
        
        // Update core std_dev to match
        self.core.std_dev = variance_estimate.sqrt();
        self.core.mean2 = variance_estimate; // For consistency
    }

    /// Reset but preserve current mean as prior
    pub fn reset_with_current_as_prior(&mut self) {
        let current_mean = self.core.mean;
        let current_variance = self.stable_variance;
        self.set_prior_with_variance(current_mean, current_variance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;
    use alloc::vec::Vec;

    use float_cmp::{ApproxEq, ApproxEqUlps};
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    use rayon::prelude::*;

    type T = f64;

    #[test]
    fn it_works() {
        let mut s: Stats<f32> = Stats::new();

        let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for v in &vals {
            s.update(*v);
        }

        assert_eq!(s.count, vals.len());

        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);

        assert!(s.mean.approx_eq_ulps(&3.0, 2));
        assert!(s.std_dev.approx_eq_ulps(&1.5811388, 2));
    }

    /// Calculate the mean of a vector of values
    fn calc_mean(vals: &Vec<T>) -> T {
        let sum = vals.iter().fold(T::zero(), |acc, x| acc + *x);

        sum / T::from_usize(vals.len()).unwrap()
    }

    /// Calculate the standard deviation of a vector of values
    fn calc_std_dev(vals: &Vec<T>) -> T {
        let mean = calc_mean(vals);
        let std_dev = (vals
            .iter()
            .fold(T::zero(), |acc, x| acc + (*x - mean).powi(2))
            / T::from_usize(vals.len() - 1).unwrap())
        .sqrt();

        std_dev
    }

    /// Get the maximum value in a vector of values
    fn get_max(vals: &Vec<T>) -> T {
        let mut max = T::min_value();
        for v in vals {
            if *v > max {
                max = *v;
            }
        }
        max
    }

    /// Get the minimum value in a vector of values
    fn get_min(vals: &Vec<T>) -> T {
        let mut min = T::max_value();
        for v in vals {
            if *v < min {
                min = *v;
            }
        }
        min
    }

    #[test]
    fn stats_for_large_random_data() {
        // Define some constants
        const MEAN: T = 2.0;
        const STD_DEV: T = 3.0;
        const SEED: u64 = 42;
        const NUM_SAMPLES: usize = 10_000;

        let mut s: Stats<T> = Stats::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();

        // Generate some random data
        let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();

        // Update the stats
        random_data.iter().for_each(|v| s.update(*v));

        // Calculate the mean using sum/count method
        let mean = calc_mean(&random_data);

        // Check the mean value against the stats' mean value
        assert!(s.mean.approx_eq(mean, (1.0e-13, 2)));

        // Calculate the standard deviation
        let std_dev = calc_std_dev(&random_data);

        // Check the standard deviation against the stats' standard deviation
        assert!(s.std_dev.approx_eq(std_dev, (1.0e-13, 2)));

        // Check the count
        assert_eq!(s.count, random_data.len());

        // Find the max and min values
        let max = get_max(&random_data);
        let min = get_min(&random_data);

        // Check the max and min values
        assert_eq!(s.max, max);
        assert_eq!(s.min, min);
    }

    #[test]
    fn stats_merge() {
        // Define some constants
        const MEAN: T = 2.0;
        const STD_DEV: T = 3.0;
        const SEED: u64 = 42;
        const NUM_SAMPLES: usize = 10_000;

        let mut s: Stats<T> = Stats::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();

        // Generate some random data
        let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();

        // Update the stats
        random_data.iter().for_each(|v| s.update(*v));

        // Calculate the stats using the aggregate method instead of the rolling method
        let mean = calc_mean(&random_data);
        let std_dev = calc_std_dev(&random_data);
        let max = get_max(&random_data);
        let min = get_min(&random_data);

        let chunks_size = 1000;

        let stats: Vec<Stats<T>> = random_data
            .chunks(chunks_size)
            .map(|chunk| {
                let mut s: Stats<T> = Stats::new();
                chunk.iter().for_each(|v| s.update(*v));
                s
            })
            .collect();

        assert_eq!(stats.len(), NUM_SAMPLES / chunks_size);

        // Accumulate the stats
        let merged_stats = stats.into_iter().reduce(|acc, s| acc.merge(&s)).unwrap();

        // Check the stats against the aggregate stats (using sum/count method)
        assert!(merged_stats.mean.approx_eq(mean, (1.0e-13, 2)));
        assert!(merged_stats.std_dev.approx_eq(std_dev, (1.0e-13, 2)));
        assert_eq!(merged_stats.max, max);
        assert_eq!(merged_stats.min, min);
        assert_eq!(merged_stats.count, NUM_SAMPLES);

        // Check the stats against the merged stats object
        assert!(merged_stats.mean.approx_eq(s.mean, (1.0e-13, 2)));
        assert!(merged_stats.std_dev.approx_eq(s.std_dev, (1.0e-13, 2)));
        assert_eq!(merged_stats.max, s.max);
        assert_eq!(merged_stats.min, s.min);
        assert_eq!(merged_stats.count, s.count);

        // Check edge cases

        // Check merging with an empty stats object
        let empty_stats: Stats<T> = Stats::new();
        let merged_stats = s.merge(&empty_stats);
        assert_eq!(merged_stats.count, s.count);

        // Check merging an empty stats object with a non-empty stats object
        let empty_stats: Stats<T> = Stats::new();
        let merged_stats = empty_stats.merge(&s);
        assert_eq!(merged_stats.count, s.count);

        // Check merging two empty stats objects
        let empty_stats_1: Stats<T> = Stats::new();
        let empty_stats_2: Stats<T> = Stats::new();

        let merged_stats = empty_stats_1.merge(&empty_stats_2);
        assert_eq!(merged_stats.count, 0);
    }

    #[test]
    fn stats_merge_parallel() {
        // Define some constants
        const MEAN: T = 2.0;
        const STD_DEV: T = 3.0;
        const SEED: u64 = 42;
        const NUM_SAMPLES: usize = 500_000;

        let mut s: Stats<T> = Stats::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();

        // Generate some random data
        let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();

        // Update the stats
        random_data.iter().for_each(|v| s.update(*v));

        // Calculate the stats using the aggregate method instead of the rolling method
        let mean = calc_mean(&random_data);
        let std_dev = calc_std_dev(&random_data);
        let max = get_max(&random_data);
        let min = get_min(&random_data);

        let chunks_size = 1000;

        let stats: Vec<Stats<T>> = random_data
            .par_chunks(chunks_size) // <--- Parallelization by Rayon
            .map(|chunk| {
                let mut s: Stats<T> = Stats::new();
                chunk.iter().for_each(|v| s.update(*v));
                s
            })
            .collect();

        // There should be more than one stat
        assert!(stats.len() >= NUM_SAMPLES / chunks_size);

        // Accumulate the stats
        let merged_stats = stats.into_iter().reduce(|acc, s| acc.merge(&s)).unwrap();

        // Check the stats against the aggregate stats (using sum/count method)
        assert!(merged_stats.mean.approx_eq(mean, (1.0e-13, 2)));
        assert!(merged_stats.std_dev.approx_eq(std_dev, (1.0e-13, 2)));
        assert_eq!(merged_stats.max, max);
        assert_eq!(merged_stats.min, min);
        assert_eq!(merged_stats.count, NUM_SAMPLES);

        // Check the stats against the merged stats object
        assert!(merged_stats.mean.approx_eq(s.mean, (1.0e-13, 2)));
        assert!(merged_stats.std_dev.approx_eq(s.std_dev, (1.0e-13, 2)));
        assert_eq!(merged_stats.max, s.max);
        assert_eq!(merged_stats.min, s.min);
        assert_eq!(merged_stats.count, s.count);
    }
}
