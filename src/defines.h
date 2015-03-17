
#ifndef DEFINES_H_H
#define DEFINES_H_H


#if defined WIN32 || defined _WIN32 || defined WINCE  
#  define VS_EXPORTS __declspec(dllexport)
#else
#  define VS_EXPORTS
#endif


#endif