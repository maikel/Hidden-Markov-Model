/*
 * Copyright 2016 Maikel Nadolski
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* --------------------------------------------------

  Test program to
 (1) compare the performance of classic iostreams,
  standard iostreams, and C-style stdio for output, and
 (2) test any overhead of sync_with_stdio(true). Standard
 iostreams by default are synchronized with stdio streams;
 the opposite was true of classic iostreams.

 optional command line argument:
 - how many numbers to output (default 1,000,000)
 - name of output file (default cout)

  When compiling, define CLASSIC or STDIO to enable
 those options; otherwise the default is to use
 standard iostreams.
 --------------------------------------------------*/
#if defined (STDIO)
   #include <stdio.h>
   #include <stdlib.h>
#elif defined (CLASSIC)
   #include <iostream>
   #include <fstream>
   #include <iomanip>
  using namespace std;
#else
   #include <iostream>                  // use standard iostreams
   #include <fstream>
   using namespace std;
#endif

#include <vector>
#include <ctime>



//=============================================================================
// struct to hold identifier and elapsed time
struct T {
    const char* s;
    double t;

    T(const char* ss, double tt) : s(ss), t(tt) {}
    T() : s(0), t(0) {}
};



int main (int argc, char *argv[])
{
    const int n = (1 < argc) ? atoi(argv[1]) : 1000000;  // number of
                                                           // iterations

#if defined( STDIO )
    FILE * target;
    target = stdout;
    if (2 < argc) {  // place output in file
        target = fopen( argv[2], "w" );
    }
#else                                         // for both iostreams libs
    ofstream target;
    ostream* op = &cout;
    if (2 < argc) {  // place output in file
        target.open(argv[2]);
        op = &target;
    }
    ostream& out = *op;
#endif

    int i;                              // for-loop variable

                                        // output command for documentation:
#if defined( STDIO )
    for (i = 0; i < argc; ++i)
        fprintf( target, "%s ", argv[i]) ;
    fprintf( target, "\n" );
#else
    for (i = 0; i < argc; ++i)
        out << argv[i] << " ";
    out << "\n";
#endif

    std::vector<T> v;                        // holds elapsed time of the tests

#if !defined( STDIO)
    #if defined (CLASSIC)
    // non-synchronized I/O is the default
    #else
    out.sync_with_stdio (false);       // must be called before any output
    #endif
#endif

    // seed the random number generator
    srand( clock() );
    clock_t t = clock();
    if (t == clock_t(-1))
    {
#if defined( STDIO )
        fprintf( stderr, "sorry, no clock\n" );
#else
        cerr << "sorry, no clock\n";
#endif
        exit(1);
    }


#if defined( STDIO )
    t = clock();
    for (i = 0; i != n; ++i)
    {
       fprintf ( target, "%d ", i);
    }
    v.push_back(T("output integers to stdio                      ", clock() - t));

    t = clock();
    for ( i = 0; i != n; ++i)
    {
       fprintf ( target, "%x ", i);
    }
    v.push_back(T("output hex integers to stdio                  ", clock() - t));

    if (clock() == clock_t(-1))
    {
        fprintf ( stderr, "sorry, clock overflow\n" );
        exit(2);
    }

    // output results
    fprintf ( stderr, "\n" );
    for (i = 0; static_cast<size_t>(i)<v.size(); i++)
        fprintf( stderr, "%s :\t%f seconds\n", v[i].s, v[i].t /CLOCKS_PER_SEC );

#else
    t = clock();
    for ( i = 0; i != n; ++i)
    {
            out << i << ' ';
    }
    v.push_back(T("output integers (sync = false)       ", clock() - t));

    out << hex;
    t = clock();
    for ( i = 0; i != n; ++i)
    {
            out << i << ' ';
    }
    v.push_back(T("output hex integers (sync = false)   ", clock() - t));

    #if defined (CLASSIC)
    out.sync_with_stdio();             // synchronize -- no argument needed
    #else
    out.sync_with_stdio (true);
    #endif

    out << dec;
    t = clock();
    for ( i = 0; i != n; ++i)
    {
            out << i << ' ';
    }
    v.push_back(T("output integers (sync = true)        ", clock() - t));

    out << hex;
    t = clock();
    for ( i = 0; i != n; ++i)
    {
            out << i << ' ';
    }
    v.push_back(T("output hex integers (sync = true)     ", clock() - t));

    if (clock() == clock_t(-1))
    {
        cerr << "sorry, clock overflow\n";
        exit(2);
    }

    // output results
    cerr << endl;
    for (i = 0; static_cast<size_t>(i) < v.size(); i++)
        cerr << v[i].s << " :\t"
            << v[i].t /CLOCKS_PER_SEC
            << " seconds" << endl;
#endif

    return 0;

}


