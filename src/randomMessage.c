#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#define MSG_FILE "input.in"

void rand_str(char *dest, size_t length) {
    char charset[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    
    while (length-- > 0) {
        size_t index = (double) rand() / RAND_MAX * (sizeof charset - 1);
        *dest++ = charset[index];
    }
}

int main(int argc, char** argv){
  if(argc != 2){
    fprintf(stderr, "Usage ./randomMessage <length> \n");
    exit(1);
  }

  size_t length = atoi(argv[1]);
  printf("Message Length: %d\n", length);

  char* dest = malloc(length * sizeof(char));
  rand_str(dest, length);

  FILE* msg = fopen(MSG_FILE, "w");
  if(msg == NULL){
    fprintf(stderr, "Unable to open the message data file\n");
    exit(1);
  }
  fwrite(dest, length, 1, msg);
  fclose(msg);

  return 0;
}
