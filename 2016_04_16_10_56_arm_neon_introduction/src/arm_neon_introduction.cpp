/**************************************************/
//Created by allen hong 
//LICENSE : Do what u want
//Fork me on Github => https://github.com/xylcbd
/**************************************************/
//std headers
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <random>
//project headers
#ifdef WIN32
#include "NEONvsSSE.h"
#elif defined(__ARM_NEON__)
#include "arm_neon.h"
#else
#error "unknown platform!"
#endif
#include "helper_functions.h"

static void fill_random_value(std::vector<float>& vec_data)
{
	std::uniform_real_distribution<float> distribution(
		std::numeric_limits<float>::min(),
		std::numeric_limits<float>::max());
	std::default_random_engine generator;

	std::generate(vec_data.begin(), vec_data.end(), [&]() { return distribution(generator); });
}
static bool is_equals_vector(const std::vector<float>& vec_a, const std::vector<float>& vec_b)
{
	if (vec_a.size() != vec_b.size())
	{
		return false;
	}
	for (size_t i = 0; i < vec_a.size(); i++)
	{
		if (vec_a[i] != vec_b[i])
		{
			return false;
		}
	}
	return true;
}
static void normal_vector_mul(const std::vector<float>& vec_a, const std::vector<float>& vec_b, std::vector<float>& vec_result)
{
	assert(vec_a.size() == vec_b.size());
	assert(vec_a.size() == vec_result.size());
	//compiler may optimized auto tree vectorize (test this diabled -ftree-vectorize)
	for (size_t i = 0; i < vec_result.size();i++)
	{
		vec_result[i] = vec_a[i] * vec_b[i];
	}
}
static void neon_vector_mul(const std::vector<float>& vec_a, const std::vector<float>& vec_b, std::vector<float>& vec_result)
{
	assert(vec_a.size() == vec_b.size());
	assert(vec_a.size() == vec_result.size());
	int i = 0;
	//neon process
	for (; i < (int)vec_result.size() - 3 ; i+=4)
	{
		const auto data_a = vld1q_f32(&vec_a[i]);
		const auto data_b = vld1q_f32(&vec_b[i]);
		float* dst_ptr = &vec_result[i];
		const auto data_res = vmulq_f32(data_a, data_b);
		vst1q_f32(dst_ptr, data_res);
	}
	//normal process
	for (; i < (int)vec_result.size(); i++)
	{
		vec_result[i] = vec_a[i] * vec_b[i];
	}
}
static int test_neon()
{
	const int test_round = 1000;
	const int data_len = 100000;
	std::vector<float> vec_a(data_len);
	std::vector<float> vec_b(data_len);
	std::vector<float> vec_result(data_len);
	std::vector<float> vec_result2(data_len);
	//fill random value in vecA & vecB
	std::cout << "round of test is " << test_round << std::endl;
	std::cout << "size of input vector is "<< data_len << std::endl;
	std::cout << "filling random data to input vector..." << std::endl;
	fill_random_value(vec_a);
	fill_random_value(vec_b);
	std::cout << "fill random data to input vector done.\n" << std::endl;
	//check the result is same
	{
		normal_vector_mul(vec_a, vec_b, vec_result);
		neon_vector_mul(vec_a, vec_b, vec_result2);
		if (!is_equals_vector(vec_result,vec_result2))
		{
			std::cerr << "result vector is not equals!" << std::endl;
			return -1;
		}
	}
	//test normal_vector_mul
	{
		FuncCostTimeHelper time_helper("normal_vector_mul");
		for (int i = 0; i < test_round;i++)
		{
			normal_vector_mul(vec_a, vec_b, vec_result);
		}
	}
	//test neon_vector_mul
	{
		FuncCostTimeHelper time_helper("neon_vector_mul");
		for (int i = 0; i < test_round; i++)
		{
			neon_vector_mul(vec_a, vec_b, vec_result2);
		}
	}
	return 0;
}

int main(int, char*[])
{
	return test_neon();
}