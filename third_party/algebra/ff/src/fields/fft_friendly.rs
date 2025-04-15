/// The interface for fields that are able to be used in FFTs.
pub trait FftField: crate::Field {
    /// The generator of the multiplicative group of the field
    const GENERATOR: Self;

    /// Let `N` be the size of the multiplicative group defined by the field.
    /// Then `TWO_ADICITY` is the two-adicity of `N`, i.e. the integer `s`
    /// such that `N = 2^s * t` for some odd integer `t`.
    const TWO_ADICITY: u32;

    /// 2^s root of unity computed by GENERATOR^t
    const TWO_ADIC_ROOT_OF_UNITY: Self;

    /// An integer `b` such that there exists a multiplicative subgroup
    /// of size `b^k` for some integer `k`.
    const SMALL_SUBGROUP_BASE: Option<u32> = None;

    /// The integer `k` such that there exists a multiplicative subgroup
    /// of size `Self::SMALL_SUBGROUP_BASE^k`.
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = None;

    /// GENERATOR^((MODULUS-1) / (2^s *
    /// SMALL_SUBGROUP_BASE^SMALL_SUBGROUP_BASE_ADICITY)) Used for mixed-radix
    /// FFT.
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<Self> = None;

    /// Returns the root of unity of order n, if one exists.
    /// If no small multiplicative subgroup is defined, this is the 2-adic root
    /// of unity of order n (for n a power of 2).
    /// If a small multiplicative subgroup is defined, this is the root of unity
    /// of order n for the larger subgroup generated by
    /// `FftConfig::LARGE_SUBGROUP_ROOT_OF_UNITY`
    /// (for n = 2^i * FftConfig::SMALL_SUBGROUP_BASE^j for some i, j).
    fn get_root_of_unity(n: u64) -> Option<Self> {
        let mut omega: Self;
        if let Some(large_subgroup_root_of_unity) = Self::LARGE_SUBGROUP_ROOT_OF_UNITY {
            let q = Self::SMALL_SUBGROUP_BASE.expect(
                "LARGE_SUBGROUP_ROOT_OF_UNITY should only be set in conjunction with SMALL_SUBGROUP_BASE",
            ) as u64;
            let small_subgroup_base_adicity = Self::SMALL_SUBGROUP_BASE_ADICITY.expect(
                "LARGE_SUBGROUP_ROOT_OF_UNITY should only be set in conjunction with SMALL_SUBGROUP_BASE_ADICITY",
            );

            let q_adicity = crate::utils::k_adicity(q, n);
            let q_part = q.checked_pow(q_adicity)?;

            let two_adicity = crate::utils::k_adicity(2, n);
            let two_part = 2u64.checked_pow(two_adicity)?;

            if n != two_part * q_part
                || (two_adicity > Self::TWO_ADICITY)
                || (q_adicity > small_subgroup_base_adicity)
            {
                return None;
            }

            omega = large_subgroup_root_of_unity;
            for _ in q_adicity..small_subgroup_base_adicity {
                omega = omega.pow([q as u64]);
            }

            for _ in two_adicity..Self::TWO_ADICITY {
                omega.square_in_place();
            }
        } else {
            // Compute the next power of 2.
            let size = n.next_power_of_two() as u64;
            let log_size_of_group = ark_std::log2(usize::try_from(size).expect("too large"));

            if n != size || log_size_of_group > Self::TWO_ADICITY {
                return None;
            }

            // Compute the generator for the multiplicative subgroup.
            // It should be 2^(log_size_of_group) root of unity.
            omega = Self::TWO_ADIC_ROOT_OF_UNITY;
            for _ in log_size_of_group..Self::TWO_ADICITY {
                omega.square_in_place();
            }
        }
        Some(omega)
    }
}
