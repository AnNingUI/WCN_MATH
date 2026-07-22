#include "WCN/Rtype.h"
#include "WCN/WCN_MATH_DST.h"
#include "WCN/WCN_MATH_CPP_COMMON.hpp"
#include "WCN/WCN_MATH_SOA.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
namespace WCN {
    template<typename T, size_t N>
    class ArrayView {
    private:
        T* data_;

    public:
        explicit ArrayView(T* data) : data_(data) {}

        T& operator[](size_t index) {
            return data_[index];
        }

        const T& operator[](size_t index) const {
            return data_[index];
        }

        size_t size() const { return N; }
    };
    template<typename T, size_t N>
    std::ostream& operator<<(std::ostream& os, const ArrayView<T, N>& view) {
        os << "[";
        for (size_t i = 0; i < N; ++i) {
            os << view[i];
            if (i != N - 1) os << ", ";
        }
        os << "]";
        return os;
    }

    // 堆版本只有 RAII 和 c_ptr()
    template<size_t N>
    struct Vec2xNDeleter {
        void operator()(WMATH_TYPE(Vec2xN)* p) const {
            wcn_math_Vec2xN_free(p);
        }
    };
    template<size_t N>
    using Vec2xNUniquePtr = std::unique_ptr<WMATH_TYPE(Vec2xN), Vec2xNDeleter<N>>;
    template<size_t N>
    inline Vec2xNUniquePtr<N> make_vec2xn_h() {
        return Vec2xNUniquePtr<N>(wcn_math_Vec2xN_malloc(N));
    }

    // 栈版本
    // 编译期计算对齐后的大小（确保16字节对齐）
    template <size_t N>
    class Vec2xN {
    public:
        using SelfType = Vec2xN<N>;
        static constexpr size_t Count = N;

    private:
        alignas(16) f32 x_data_[N]{};
        alignas(16) f32 y_data_[N]{};

        WMATH_TYPE(Vec2xN) c_wrapper_{};

    public:
        // 默认构造
        Vec2xN() {
            c_wrapper_.count = N;
            c_wrapper_.x = x_data_;
            c_wrapper_.y = y_data_;
        }

        // 从 std::array 或 { } 构造
        Vec2xN(const std::array<f32, N>& x, const std::array<f32, N>& y) : Vec2xN() {
            std::copy(x.begin(), x.end(), x_data_);  // 批量拷贝x
            std::copy(y.begin(), y.end(), y_data_);  // 批量拷贝y
        }

        // 自定义拷贝构造/赋值：确保 c_wrapper_ 指向本对象的数据
        Vec2xN(const SelfType& other) {
            std::copy(other.x_data_, other.x_data_ + N, x_data_);
            std::copy(other.y_data_, other.y_data_ + N, y_data_);
            c_wrapper_.count = N;
            c_wrapper_.x = x_data_;
            c_wrapper_.y = y_data_;
        }

        Vec2xN& operator=(const SelfType& other) {
            if (this == &other) return *this;
            std::copy(other.x_data_, other.x_data_ + N, x_data_);
            std::copy(other.y_data_, other.y_data_ + N, y_data_);
            c_wrapper_.count = N;
            c_wrapper_.x = x_data_;
            c_wrapper_.y = y_data_;
            return *this;
        }

        // 移动构造/赋值（复制数据并清空源）
        Vec2xN(SelfType&& other) noexcept {
            std::copy(other.x_data_, other.x_data_ + N, x_data_);
            std::copy(other.y_data_, other.y_data_ + N, y_data_);
            other.clear();
            c_wrapper_.count = N;
            c_wrapper_.x = x_data_;
            c_wrapper_.y = y_data_;
        }

        SelfType& operator=(SelfType&& other) noexcept {
            if (this == &other) return *this;
            std::copy(other.x_data_, other.x_data_ + N, x_data_);
            std::copy(other.y_data_, other.y_data_ + N, y_data_);
            other.clear();
            c_wrapper_.count = N;
            c_wrapper_.x = x_data_;
            c_wrapper_.y = y_data_;
            return *this;
        }

        ~Vec2xN() = default;

        // 清空
        void clear() {
            std::fill(x_data_, x_data_ + N, 0.0f);
            std::fill(y_data_, y_data_ + N, 0.0f);
        }

        // ArrayView 访问
        ArrayView<f32, N> x() { return ArrayView<f32, N>(x_data_); }
        ArrayView<f32, N> y() { return ArrayView<f32, N>(y_data_); }
        ArrayView<const f32, N> x() const { return ArrayView<const f32, N>(x_data_); }
        ArrayView<const f32, N> y() const { return ArrayView<const f32, N>(y_data_); }

        // C 指针
        WMATH_TYPE(Vec2xN)* c_ptr() { return &c_wrapper_; }
        const WMATH_TYPE(Vec2xN)* c_ptr() const { return &c_wrapper_; }

        // ---------------- 运算符 ----------------
        SelfType operator+(const SelfType& other) const {
            SelfType result;
            WMATH_ADD(Vec2xN)(result.c_ptr(), this->c_ptr(), other.c_ptr());
            return result;
        }

        SelfType& operator+=(const SelfType& other) {
            WMATH_ADD(Vec2xN)(this->c_ptr(), this->c_ptr(), other.c_ptr());
            return *this;
        }

        SelfType operator-(const SelfType& other) const {
            SelfType result;
            WMATH_SUB(Vec2xN)(result.c_ptr(), this->c_ptr(), other.c_ptr());
            return result;
        }

        SelfType add_scaled(const SelfType& other, f32 s) const {
            SelfType result;
            WMATH_ADD_SCALED(Vec2xN)(result.c_ptr(), this->c_ptr(), other.c_ptr(), s);
            return result;
        }

        // ---------------- 单个元素操作 ----------------
        bool set(size_t index, f32 x, f32 y) {
            if (index >= N) return false;
            x_data_[index] = x;
            y_data_[index] = y;
            return true;
        }

        bool get(size_t index, f32& x, f32& y) const {
            if (index >= N) return false;
            x = x_data_[index];
            y = y_data_[index];
            return true;
        }
    };

    // ===================== 性能测试工具函数 =====================
    static volatile f32 g_benchmark_sink = 0.0f; // 防止优化掉的接收器

    // 计时工具：返回毫秒数
    template<typename Func>
    double measure_time(Func&& func, int iterations = 1000000) {
        using namespace std::chrono;
        
        // 预热（避免首次执行的缓存/初始化影响）
        func();
        
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        auto end = high_resolution_clock::now();
        
        duration<double, std::milli> duration = end - start;
        return duration.count() / iterations;  // 单次执行耗时（毫秒）
    }

    // 通用性能测试函数（纯英文输出，避免编码问题）
    template<size_t N>
    void run_performance_test(int iterations = 100000) {
        std::cout << "\n=== Vec2xN Performance Test (N=" << N << ", Iterations=" << iterations << ") ===\n";
        
        // 准备测试数据
        std::array<f32, N> x_data, y_data;
        for (size_t i = 0; i < N; ++i) {
            x_data[i] = static_cast<f32>(i) / N;
            y_data[i] = static_cast<f32>(i * 2) / N;
        }

        // ---------------- 栈版本性能测试 ----------------
        std::cout << "2. Stack Version (Vec2xN) Test:\n";
        
        // 加法运算性能
        double stack_add_time = measure_time([&]() {
            Vec2xN<N> a(x_data, y_data);
            Vec2xN<N> b(x_data, y_data);
            auto c = a + b;
            g_benchmark_sink = c.x()[0];
        }, iterations);
        std::cout << "   Addition operation time: " << std::fixed << std::setprecision(6) << stack_add_time << " ms/iteration\n";

        // 赋值加法性能
        double stack_add_assign_time = measure_time([&]() {
            Vec2xN<N> a(x_data, y_data);
            Vec2xN<N> b(x_data, y_data);
            a += b;
            g_benchmark_sink = a.x()[0];
        }, iterations);
        std::cout << "   Addition assignment time: " << std::fixed << std::setprecision(6) << stack_add_assign_time << " ms/iteration\n";
    }

    // 通用迭代器类模板
    // 通用迭代器类模板
    template<size_t N>
    class Vec2xNIterator {
    private:
        const WMATH_TYPE(Vec2xN)* container_; // 直接存储底层数据指针
        size_t index_;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Vec2;
        using difference_type = std::ptrdiff_t;
        using reference = Vec2; // 返回值语义

        Vec2xNIterator(const WMATH_TYPE(Vec2xN)* container, size_t index)
            : container_(container), index_(index) {}

        Vec2 operator*() const {
            Vec2 v;
            wcn_math_Vec2xN_get(&v, container_, index_);
            return v;
        }

        Vec2xNIterator& operator++() {
            ++index_;
            return *this;
        }

        bool operator!=(const Vec2xNIterator& other) const {
            return index_ != other.index_;
        }
    };

    // 为栈版本 Vec2xN<N> 添加 begin/end
    template<size_t N>
    Vec2xNIterator<N> begin(Vec2xN<N>& vec2xn) {
        return Vec2xNIterator<N>(vec2xn.c_ptr(), 0);
    }

    template<size_t N>
    Vec2xNIterator<N> end(Vec2xN<N>& vec2xn) {
        return Vec2xNIterator<N>(vec2xn.c_ptr(), N);
    }

    // 为堆版本 Vec2xNUniquePtr<N> 添加 begin/end
    template<size_t N>
    Vec2xNIterator<N> begin(Vec2xNUniquePtr<N>& vec2xn) {
        return Vec2xNIterator<N>(vec2xn.get(), 0);
    }

    template<size_t N>
    Vec2xNIterator<N> end(Vec2xNUniquePtr<N>& vec2xn) {
        return Vec2xNIterator<N>(vec2xn.get(), N);
    }

    // 为 const 堆版本 Vec2xNUniquePtr<N> 添加 begin/end
    template<size_t N>
    Vec2xNIterator<N> begin(const Vec2xNUniquePtr<N>& vec2xn) {
        return Vec2xNIterator<N>(vec2xn.get(), 0);
    }

    template<size_t N>
    Vec2xNIterator<N> end(const Vec2xNUniquePtr<N>& vec2xn) {
        return Vec2xNIterator<N>(vec2xn.get(), N);
    }
}
static const auto g_vec2xn = WCN::make_vec2xn_h<4>();

int main() {
    {
        wcn_math_Vec2xN_set_by_xy(g_vec2xn.get(), 0, 1.0f, 2.0f);
        wcn_math_Vec2xN_set_by_xy(g_vec2xn.get(), 1, 2.0f, 3.0f);
        wcn_math_Vec2xN_set_by_xy(g_vec2xn.get(), 2, 3.0f, 4.0f);
        wcn_math_Vec2xN_set_by_xy(g_vec2xn.get(), 3, 4.0f, 5.0f);
    }
    for (const auto& v : g_vec2xn) {
        std::cout << "Heap Vec2xN: (" << v.v[0] << ", " << v.v[1] << ")\n";
    }
    // in c
    WCN::Vec2 v1;
    WCN::Vec2 v2{1, 2};
    WCN::Vec2 v3{2, 3};
    WMATH_ADD(Vec2)(&v1, v2, v3);
    
    std::cout << "v1.x = " << v1.v[0] << "\n" << "v1.y = " << v1.v[1] << "\n";
    // auto v4 = wcn_math_Vec2xN_malloc(4);
    // wcn_math_Vec2xN_set_by_xy(v4, 0, 1, 2);
    // wcn_math_Vec2xN_set_by_xy(v4, 1, 2, 3);
    WCN::Vec2xN<4> v4{
        {1, 2, 3, 4},
        {2, 3, 4, 5}
    };
    WCN::Vec2xN<4> v5{
        {1, 2, 3, 4},
        {2, 3, 4, 5}
    };
    WCN::Vec2xN<4> v6 = v4 + v5;
    std::cout << "v6.x = " << v6.x() << "\n" << "v6.y = " << v6.y() << "\n";
    for (const auto& v : v6) {
        std::cout << "Stack Vec2xN: (" << v.v[0] << ", " << v.v[1] << ")\n";
    }
    // 性能测试
    // 测试不同大小的N值
    WCN::run_performance_test<4>(1000000);    // 小数据量
    WCN::run_performance_test<16>(1000000);   // 中等数据量
    WCN::run_performance_test<64>(100000);    // 大数据量（减少迭代次数）
}