#ifndef BUFFERED_BINARY_ITERATOR_H_
#define BUFFERED_BINARY_ITERATOR_H_

#include <iterator>

#include "gsl_assert.h"

namespace mnb {

  template <class _Tp>
  using alpha_type = std::pair<_Tp, std::vector<_Tp>>;

  template <typename _Tp, typename _CharT = char,
      typename _Traits = std::char_traits<_CharT>, typename _Dist = ptrdiff_t>
  class alphas_binary_input_iterator
  : public std::iterator<std::input_iterator_tag,
      alpha_type<_Tp>, _Dist, alpha_type<_Tp>*, alpha_type<_Tp>&>
  {
    public:
      typedef _CharT                         char_type;
      typedef _Traits                        traits_type;
      typedef std::basic_istream<_CharT, _Traits> istream_type;

    private:
      istream_type*         _M_stream = nullptr;
      std::size_t           _M_num_states = 0;
      alpha_type<_Tp>       _M_alpha;
      bool                  _M_ok = false;

    public:
      alphas_binary_input_iterator() = default;
//      alphas_binary_input_iterator(const alphas_binary_input_iterator& other)
//      : _M_stream(other._M_stream), _M_num_states(other._M_num_states),
//        _M_alpha(other._M_alpha), _M_ok(other._M_ok) { }

      alphas_binary_input_iterator(
          istream_type& stream, std::size_t num_states)
      : _M_stream(std::addressof(stream)), _M_num_states(num_states), _M_ok(true)
      {
        Expects(_M_num_states > 0);
        _M_alpha.second = std::vector<_Tp>(_M_num_states);
        _M_read();
      }


      const alpha_type<_Tp>&
      operator*() const
      {
        Expects(_M_ok);
        return _M_alpha;
      }

      const _Tp*
      operator->() const { return &(operator*()); }

      alphas_binary_input_iterator&
      operator++()
      {
        Expects(_M_ok);
        _M_read();
        return *this;
      }

      alphas_binary_input_iterator
      operator++(int)
      {
        Expects(_M_ok);
        alphas_binary_input_iterator __tmp = *this;
        _M_read();
        return __tmp;
      }

      bool
      _M_equal(const alphas_binary_input_iterator& __x) const
      { return (_M_ok == __x._M_ok) && (!_M_ok || _M_stream == __x._M_stream); }

    private:
      void _M_read()
      {
        _M_ok = (_M_stream != nullptr && *_M_stream) ? true : false;
        if (_M_ok) {
          std::size_t type_size = sizeof(_Tp);
          std::size_t buffer_size = type_size*(_M_num_states+1);
          std::vector<char_type> buffer(buffer_size);
          _M_stream->read(buffer.data(), buffer_size);
          _M_ok = *_M_stream ? true : false;
          if (_M_ok && _M_stream->gcount()
              == gsl::narrow<std::streamsize>(buffer_size)) {
            std::copy(buffer.data(), buffer.data()+type_size,
                reinterpret_cast<char_type*>(&_M_alpha.first));
            std::copy(
                buffer.data()+type_size,
                buffer.data()+buffer_size,
                reinterpret_cast<char_type*>(_M_alpha.second.data()));
          }
        }
      }
  };

  ///  Return true if x and y are both end or not end, or x and y are the same.
  template<typename _Tp, typename _CharT, typename _Traits, typename _Dist>
  inline bool
  operator==(const alphas_binary_input_iterator<_Tp, _CharT, _Traits, _Dist>& __x,
             const alphas_binary_input_iterator<_Tp, _CharT, _Traits, _Dist>& __y)
  { return __x._M_equal(__y); }

  ///  Return false if x and y are both end or not end, or x and y are the same.
  template <class _Tp, class _CharT, class _Traits, class _Dist>
  inline bool
  operator!=(const alphas_binary_input_iterator<_Tp, _CharT, _Traits, _Dist>& __x,
             const alphas_binary_input_iterator<_Tp, _CharT, _Traits, _Dist>& __y)
  { return !__x._M_equal(__y); }

  template <typename _Tp, std::size_t N, typename _CharT = char,
      typename _Traits = std::char_traits<_CharT>>
  class ostream_buffered_binary_iterator
  : public std::iterator<std::output_iterator_tag, void, void, void, void> {
    public:
      //@{
      /// Public typedef
      typedef _CharT                              char_type;
      typedef _Traits                             traits_type;
      typedef std::basic_ostream<_CharT, _Traits> ostream_type;
      //@}

      static const std::size_t buffer_size = N;
    private:
      ostream_type&                      _M_stream;
      std::array<char_type, buffer_size> _M_buffer;
      std::size_t                        _M_num_elements = 0;
    public:
      explicit ostream_buffered_binary_iterator(ostream_type& stream) noexcept
    : _M_stream(stream) {
        Expects(buffer_size > 0);
        Expects(stream);
      }

      ~ostream_buffered_binary_iterator() noexcept
      {
        flush();
      }

      void flush() {
        if (_M_num_elements > 0 && _M_stream)
          _M_stream.write(_M_buffer.data(), _M_num_elements);
      }

      /// Writes @a value to underlying ostream using write().
      ostream_buffered_binary_iterator&
      operator=(std::vector<_Tp> const& __value)
      {
        std::size_t num_elems = __value.size();
        std::size_t type_size = sizeof(_Tp);
        Expects(_M_num_elements <= buffer_size);
        if (buffer_size-_M_num_elements <= num_elems*type_size) {
          _M_stream.write(_M_buffer.data(), _M_num_elements);
          _M_num_elements = 0;
          char_type const* begin =
              reinterpret_cast<char_type const*>(__value.data());
          _M_stream.write(begin, num_elems*type_size);
        } else {
          char_type const* begin =
              reinterpret_cast<char_type const*>(__value.data());
          char_type const* end = begin + num_elems*type_size;
          std::copy(begin, end, _M_buffer.data() + _M_num_elements);
          _M_num_elements += num_elems*type_size;
        }
        return *this;
      }

      ostream_buffered_binary_iterator&
      operator=(_Tp const& __value)
      {
        std::size_t type_size = sizeof(_Tp);
        Expects(_M_num_elements <= buffer_size);
        if (buffer_size-_M_num_elements <= type_size) {
          _M_stream.write(_M_buffer.data(), _M_num_elements);
          _M_num_elements = 0;
          char_type const* begin =
              reinterpret_cast<char_type const*>(&__value);
          _M_stream.write(begin, type_size);
        } else {
          char_type const* begin =
              reinterpret_cast<char_type const*>(&__value);
          char_type const* end = begin + type_size;
          std::copy(begin, end, _M_buffer.data() + _M_num_elements);
          _M_num_elements += type_size;
        }
        return *this;
      }

      ostream_buffered_binary_iterator&
      operator=(std::pair<_Tp, std::vector<_Tp>> const& pair)
      {
        (*this) = pair.first;
        (*this) = pair.second;
        return *this;
      }

      ostream_buffered_binary_iterator&
      operator*()
      { return *this; }

      ostream_buffered_binary_iterator&
      operator++()
      { return *this; }

      ostream_buffered_binary_iterator&
      operator++(int)
      { return *this; }
  };


}


#endif /* BUFFERED_BINARY_ITERATOR_H_ */
