#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>

int main(int argc, char **argv)
{
    pid_t pid = fork ();
    if (pid != 0) {
        waitpid (pid, NULL, 0);
    }
    printf ("Hello, World!\n");
}