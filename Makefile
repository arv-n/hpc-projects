# Variables 

CC = gcc
CFLAGS = -g -Wall

#Targets
check-syntax:
	${CC} -o nul -S ${CHK_SOURCES}

clean:
	$(RM) helloworld *.o *~ *.dat *.txt
