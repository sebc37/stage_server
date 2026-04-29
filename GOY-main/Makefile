#####################################################################
# makefile for GOY shell model code			                        #
#####################################################################
# (c) Nicolas B. Garnier                 			                #
# last revision 2022-08-30                                          #
#####################################################################
# usage :                                                           #
#                                                                   #
# make               : to produce executable from C                 #
#####################################################################

OBJS = integrate.o integrate_io.o stats_io.o

run : $(OBJS) main.o 
	$(CC) $(OBJS) main.o -lm -o run
	mkdir -p stats
	mkdir -p tmp

%.o : %.c parameters.h
	$(CC) -Wall -Wextra -Wshadow -Wpointer-arith -Wundef -Wunreachable-code -c $<

clean :
	rm -f *.o run
