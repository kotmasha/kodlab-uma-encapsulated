#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#if defined(_WIN64)
#include <Windows.h>

#else
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
extern char **environ;

#endif

#define BUFF_SIZE 1024

long readPID() {
	char buff[BUFF_SIZE];
	FILE *f = fopen("pid.txt", "r");
	if (f == NULL) {
		return -1;
	}

	fgets(buff, BUFF_SIZE, f);
	unsigned long pid = (unsigned long)atoi(buff);

	fclose(f);
	return pid;
}

void saveToPID(unsigned long pid) {
	FILE *f = fopen("pid.txt", "w");
	if (f == NULL) {
		printf("Error opening file!\n");
		exit(1);
	}

	fprintf(f, "%d", pid);
	fclose(f);
}


int isUMARunning(long pid) {
	if (pid == -1) {
		return 0;
	}
	unsigned long upid = (unsigned long)pid;
#if defined(_WIN64)
	HANDLE h = OpenProcess(PROCESS_ALL_ACCESS, TRUE, upid);
	return !h == NULL;
#else
	return 0 == kill((pid_t) pid, 0);
#endif
}


void startUMA() {
#if defined(_WIN64)
	STARTUPINFO info = { sizeof(info) };
	PROCESS_INFORMATION processInfo;
	if (CreateProcess("UMAc.exe", NULL, NULL, NULL, TRUE, 0, NULL, NULL, &info, &processInfo))
	{
		CloseHandle(processInfo.hProcess);
		CloseHandle(processInfo.hThread);
		saveToPID(processInfo.dwProcessId);
	}
#else // on linux
	pid_t pid;
	char *argv[] = {"UMAc", (char *) 0};
	int status;
	status = posix_spawn(&pid, "./UMAc", NULL, NULL, argv, environ);
	if (status == 0) {
		saveToPID((unsigned long)pid);
		printf("Child id: %i\n", pid);
		fflush(NULL);
	} 
	else {
		printf("posix_spawn: %s\n", strerror(status));
	}
#endif
}

void stopUMA(long pid){
#if defined(_WIN64)
	HANDLE h = OpenProcess(PROCESS_ALL_ACCESS, TRUE, pid);
	if (0 != TerminateProcess(h, 0)) {
		printf("UMA is successfully terminated\n");
	}
	else {
		printf("UMA termaination failed!\n");
	}
        CloseHandle(h);
#else
	if (0 == kill((pid_t) pid, SIGTERM)) {
		printf("UMA is successfully terminated\n");
	}
	else {
		printf("UMA termaination failed!\n");
	}
#endif
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		printf("Usage: UMA.exe start|stop|status\n");
		exit(0);
	}

	char * const cmd = argv[1];
	long pid = readPID();
	if (0 == strcmp(cmd, "start")) {
		if (!isUMARunning(pid)) {
			startUMA();
		}
		else {
			printf("UMA is already running, pid=%d\n", pid);
		}
	}
	else if (0 == strcmp(cmd, "stop")) {
		if (!isUMARunning(pid)) {
			printf("UMA is not running\n");
		}
		else {
			stopUMA(pid);
		}
	}
	else if(0 == strcmp(cmd, "status")){
		if (!isUMARunning(pid)) {
			printf("UMA is not running\n");;
		}
		else {
			printf("UMA is running, pid=%d\n", pid);
		}
	}

	return 1;
}
