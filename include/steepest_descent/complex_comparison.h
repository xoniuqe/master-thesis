#pragma once

#include <complex>

namespace complex_comparison {
	template<typename T>
	constexpr bool operator >= (const std::complex<T>& l, const std::complex<T>& r)
	{
		return std::real(l) >= std::real(r);
	}
	template<typename T>
	constexpr bool operator >= (const std::complex<T>& l, const T& r)
	{
		return std::real(l) >= r;
	}

	template<typename T>
	constexpr bool operator > (const std::complex<T>& l, const std::complex<T>& r)
	{
		return std::real(l) > std::real(r);
	}
	template<typename T>
	constexpr bool operator > (const std::complex<T>& l, const T& r)
	{
		return std::real(l) > r;
	}

	template<typename T>
	constexpr bool operator <= (const std::complex<T>& l, const std::complex<T>& r)
	{
		return std::real(l) <= std::real(r);
	}
	template<typename T>
	constexpr bool operator <= (const std::complex<T>& l, const T& r)
	{
		return std::real(l) <= r;
	}


	template<typename T>
	constexpr bool operator < (const std::complex<T>& l, const std::complex<T>& r)
	{
		return std::real(l) < std::real(r);
	}
	template<typename T>
	constexpr bool operator < (const std::complex<T>& l, T& r)
	{
		return std::real(l) < r;
	}

}