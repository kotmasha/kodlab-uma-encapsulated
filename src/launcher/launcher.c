#include <stdio.h>
#include <Windows.h>

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
	HANDLE h = OpenProcess(PROCESS_ALL_ACCESS, TRUE, upid);

	return !h == NULL;
}

void startUMA() {
	STARTUPINFO info = { sizeof(info) };
	PROCESS_INFORMATION processInfo;
	if (CreateProcess("UMAc.exe", NULL, NULL, NULL, TRUE, 0, NULL, NULL, &info, &processInfo))
	{
		CloseHandle(processInfo.hProcess);
		CloseHandle(processInfo.hThread);
		saveToPID(processInfo.dwProcessId);
	}
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
			HANDLE h = OpenProcess(PROCESS_ALL_ACCESS, TRUE, pid);
			if (0 != TerminateProcess(h, 0)) {
				printf("UMA is successfully terminated\n");
			}
			else {
				printf("UMA termaination failed!\n");
			}
			CloseHandle(h);
		}
	}
	else {
		if (!isUMARunning(pid)) {
			printf("UMA is not running\n");;
		}
		else {
			printf("UMA is running, pid=%d\n", pid);
		}
	}

	return 1;
}