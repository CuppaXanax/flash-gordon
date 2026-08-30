#include "fg_fabric.h"
#include "fg_uring.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

typedef struct fabric_channel{int fd;uint32_t fixed_slot;}fabric_channel;
struct fg_fabric{uint32_t rank;fg_uring *ring;fabric_channel peer[FG_RANK_COUNT][2];};
typedef struct fabric_hello{uint32_t magic_be;uint16_t version_be;uint8_t rank;uint8_t channel;uint8_t manifest_sha256[32];}fabric_hello;

static uint16_t swap16(uint16_t x){return (uint16_t)((x>>8)|(x<<8));}
static uint64_t fabric_ns(void){struct timespec value;clock_gettime(CLOCK_REALTIME,&value);return (uint64_t)value.tv_sec*UINT64_C(1000000000)+(uint64_t)value.tv_nsec;}
static void close_fd(int *fd){if(*fd>=0){close(*fd);*fd=-1;}}
static int socket_configure(int fd){int one=1;if(setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one))||setsockopt(fd,SOL_SOCKET,SO_KEEPALIVE,&one,sizeof(one)))return -1;return 0;}
static bool parse_endpoint(const char *text,struct in_addr *addr,uint16_t *port){const char *colon=strrchr(text,':');if(!colon||colon==text)return false;char ip[64];size_t n=(size_t)(colon-text);if(n>=sizeof(ip))return false;memcpy(ip,text,n);ip[n]=0;char *end;unsigned long p=strtoul(colon+1,&end,10);if(*end||p<1024||p>65534||inet_pton(AF_INET,ip,addr)!=1)return false;*port=(uint16_t)p;return true;}
static int make_listener(struct in_addr addr,uint16_t port){int fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);if(fd<0)return -1;int one=1;setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));struct sockaddr_in sa={.sin_family=AF_INET,.sin_addr=addr,.sin_port=htons(port)};if(bind(fd,(struct sockaddr *)&sa,sizeof(sa))||listen(fd,FG_RANK_COUNT)){close(fd);return -1;}return fd;}
static int connect_retry(struct in_addr addr,uint16_t port){struct sockaddr_in sa={.sin_family=AF_INET,.sin_addr=addr,.sin_port=htons(port)};struct timespec pause={.tv_sec=0,.tv_nsec=200000000};for(unsigned attempt=0;attempt<1800;attempt++){int fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);if(fd<0)return -1;if(connect(fd,(struct sockaddr *)&sa,sizeof(sa))==0){if(socket_configure(fd)==0)return fd;close(fd);return -1;}int saved=errno;close(fd);if(saved!=ECONNREFUSED&&saved!=EINTR&&saved!=ETIMEDOUT&&saved!=EHOSTUNREACH){errno=saved;return -1;}nanosleep(&pause,NULL);}errno=ETIMEDOUT;return -1;}
static bool sync_all(int fd,void *buf,size_t bytes,bool send_mode){uint8_t *p=buf;while(bytes){ssize_t n=send_mode?send(fd,p,bytes,MSG_NOSIGNAL):recv(fd,p,bytes,MSG_WAITALL);if(n<0&&errno==EINTR)continue;if(n<=0)return false;p+=n;bytes-=(size_t)n;}return true;}
static void hello_fill(fabric_hello *h,const fg_manifest *m,uint32_t rank,fg_fabric_class cls){memset(h,0,sizeof(*h));h->magic_be=htonl(FG_FRAME_MAGIC);h->version_be=swap16(FG_PROTOCOL_VERSION);h->rank=(uint8_t)rank;h->channel=(uint8_t)cls;memcpy(h->manifest_sha256,m->manifest_sha256,32);}
static bool hello_valid(const fabric_hello *h,const fg_manifest *m,uint32_t rank,fg_fabric_class cls){return ntohl(h->magic_be)==FG_FRAME_MAGIC&&swap16(h->version_be)==FG_PROTOCOL_VERSION&&h->rank==rank&&h->channel==(uint8_t)cls&&memcmp(h->manifest_sha256,m->manifest_sha256,32)==0;}
static fg_status outgoing(const fg_manifest *m,uint32_t self,uint32_t peer,fg_fabric_class cls,int *out,fg_error *err){struct in_addr addr;uint16_t port;if(!parse_endpoint(m->ranks[peer].endpoint,&addr,&port)){fg_error_set(err,FG_ERR_FORMAT,"invalid endpoint for rank %u",peer);return FG_ERR_FORMAT;}port=(uint16_t)(port+(uint16_t)cls);int fd=connect_retry(addr,port);if(fd<0){fg_error_set(err,FG_ERR_IO,"connect rank %u channel %u: %s",peer,cls,strerror(errno));return FG_ERR_IO;}fabric_hello ours,theirs;hello_fill(&ours,m,self,cls);if(!sync_all(fd,&ours,sizeof(ours),true)||!sync_all(fd,&theirs,sizeof(theirs),false)||!hello_valid(&theirs,m,peer,cls)){close(fd);fg_error_set(err,FG_ERR_MISMATCH,"rank %u handshake mismatch",peer);return FG_ERR_MISMATCH;}*out=fd;return FG_OK;}
static fg_status incoming_any(const fg_manifest *m,uint32_t self,fg_fabric_class cls,int listener,uint32_t *peer_out,int *out,fg_error *err){int fd=accept4(listener,NULL,NULL,SOCK_CLOEXEC);if(fd<0||socket_configure(fd)){if(fd>=0)close(fd);fg_error_set(err,FG_ERR_IO,"accept channel %u: %s",cls,strerror(errno));return FG_ERR_IO;}fabric_hello theirs,ours;if(!sync_all(fd,&theirs,sizeof(theirs),false)){close(fd);fg_error_set(err,FG_ERR_IO,"read incoming handshake");return FG_ERR_IO;}uint32_t peer=theirs.rank;if(peer<=self||peer>=FG_RANK_COUNT||!hello_valid(&theirs,m,peer,cls)){close(fd);fg_error_set(err,FG_ERR_MISMATCH,"incoming rank %u handshake mismatch",peer);return FG_ERR_MISMATCH;}hello_fill(&ours,m,self,cls);if(!sync_all(fd,&ours,sizeof(ours),true)){close(fd);fg_error_set(err,FG_ERR_IO,"send handshake response to rank %u",peer);return FG_ERR_IO;}*peer_out=peer;*out=fd;return FG_OK;}

fg_status fg_fabric_open(fg_fabric **out,const fg_manifest *m,uint32_t rank,fg_error *err){
    if(!out||!m||rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_ARGUMENT,"invalid fabric arguments");return FG_ERR_ARGUMENT;}*out=NULL;fg_fabric *f=calloc(1,sizeof(*f));if(!f){fg_error_set(err,FG_ERR_OOM,"allocate fabric");return FG_ERR_OOM;}f->rank=rank;for(uint32_t p=0;p<FG_RANK_COUNT;p++)for(uint32_t c=0;c<2;c++)f->peer[p][c].fd=-1;struct in_addr local;uint16_t port;if(!parse_endpoint(m->ranks[rank].endpoint,&local,&port)){fg_error_set(err,FG_ERR_FORMAT,"invalid local rank endpoint");free(f);return FG_ERR_FORMAT;}int listeners[2]={make_listener(local,port),make_listener(local,(uint16_t)(port+1))};if(listeners[0]<0||listeners[1]<0){fg_error_set(err,FG_ERR_IO,"listen on rank %u endpoint: %s",rank,strerror(errno));close_fd(&listeners[0]);close_fd(&listeners[1]);free(f);return FG_ERR_IO;}fg_status rc=FG_OK;
    /* Connect downward, then accept upward. Rank 0 becomes the startup root,
       which avoids a cycle where every connector waits for an ACK from a
       peer that has not entered accept yet. */
    for(uint32_t p=0;rc==FG_OK&&p<rank;p++)for(uint32_t c=0;rc==FG_OK&&c<2;c++)rc=outgoing(m,rank,p,(fg_fabric_class)c,&f->peer[p][c].fd,err);
    for(uint32_t c=0;rc==FG_OK&&c<2;c++)for(uint32_t accepted=rank+1;rc==FG_OK&&accepted<FG_RANK_COUNT;accepted++){uint32_t peer=0;int fd=-1;rc=incoming_any(m,rank,(fg_fabric_class)c,listeners[c],&peer,&fd,err);if(rc==FG_OK&&f->peer[peer][c].fd>=0){close(fd);fg_error_set(err,FG_ERR_MISMATCH,"duplicate rank %u channel %u",peer,c);rc=FG_ERR_MISMATCH;}else if(rc==FG_OK)f->peer[peer][c].fd=fd;}
    close_fd(&listeners[0]);close_fd(&listeners[1]);if(rc==FG_OK)rc=fg_uring_create(&f->ring,FG_RING_FABRIC,1024,err);for(uint32_t p=0;rc==FG_OK&&p<FG_RANK_COUNT;p++)if(p!=rank)for(uint32_t c=0;rc==FG_OK&&c<2;c++)rc=fg_uring_register_file(f->ring,f->peer[p][c].fd,&f->peer[p][c].fixed_slot,err);if(rc!=FG_OK){fg_fabric_close(f);return rc;}*out=f;return FG_OK;
}
void fg_fabric_close(fg_fabric *f){if(!f)return;fg_uring_destroy(f->ring);for(uint32_t p=0;p<FG_RANK_COUNT;p++)for(uint32_t c=0;c<2;c++)close_fd(&f->peer[p][c].fd);free(f);}
fg_status fg_fabric_send(fg_fabric *f,uint32_t peer,fg_fabric_class cls,fg_message_type type,uint64_t req,uint32_t seq,uint32_t flags,const void *payload,uint32_t bytes,fg_error *err){if(!f||peer>=FG_RANK_COUNT||peer==f->rank||cls>FG_FABRIC_BULK){fg_error_set(err,FG_ERR_ARGUMENT,"invalid fabric destination");return FG_ERR_ARGUMENT;}fg_frame_header h;fg_status rc=fg_frame_encode(&h,type,req,seq,flags,payload,bytes,err);if(rc!=FG_OK)return rc;if(bytes&&bytes<=65536u-sizeof(h)){/* Merge header+payload into single send to halve io_uring round-trips */uint8_t combined[65536u];memcpy(combined,&h,sizeof(h));memcpy(combined+sizeof(h),payload,bytes);rc=fg_uring_send_all(f->ring,f->peer[peer][cls].fixed_slot,combined,sizeof(h)+bytes,err);}else{rc=fg_uring_send_all(f->ring,f->peer[peer][cls].fixed_slot,&h,sizeof(h),err);if(rc==FG_OK&&bytes)rc=fg_uring_send_all(f->ring,f->peer[peer][cls].fixed_slot,payload,bytes,err);}return rc;}
fg_status fg_fabric_recv_timed(fg_fabric *f,uint32_t peer,fg_fabric_class cls,fg_frame_header *h,void *payload,uint32_t cap,uint32_t *bytes,fg_fabric_recv_timing *timing,fg_error *err){if(!f||peer>=FG_RANK_COUNT||peer==f->rank||cls>FG_FABRIC_BULK||!h){fg_error_set(err,FG_ERR_ARGUMENT,"invalid fabric receive");return FG_ERR_ARGUMENT;}fg_status rc=fg_uring_recv_all(f->ring,f->peer[peer][cls].fixed_slot,h,sizeof(*h),err);if(timing)timing->header_end_ns=fabric_ns();uint32_t n=rc==FG_OK?ntohl(h->bytes_be):0;if(rc==FG_OK&&(n>cap||(n&&!payload))){fg_error_set(err,FG_ERR_LIMIT,"fabric payload %u exceeds receive buffer %u",n,cap);return FG_ERR_LIMIT;}if(rc==FG_OK&&n)rc=fg_uring_recv_all(f->ring,f->peer[peer][cls].fixed_slot,payload,n,err);if(timing)timing->payload_end_ns=fabric_ns();if(rc==FG_OK)rc=fg_frame_validate(h,payload,bytes,err);if(timing)timing->validate_end_ns=fabric_ns();return rc;}
fg_status fg_fabric_recv(fg_fabric *f,uint32_t peer,fg_fabric_class cls,fg_frame_header *h,void *payload,uint32_t cap,uint32_t *bytes,fg_error *err){return fg_fabric_recv_timed(f,peer,cls,h,payload,cap,bytes,NULL,err);}
fg_status fg_fabric_recv_any_timed(fg_fabric *f,fg_fabric_class cls,uint32_t *peer,fg_frame_header *header,void *payload,uint32_t capacity,uint32_t *bytes,fg_fabric_recv_timing *timing,fg_error *err){if(!f||cls>FG_FABRIC_BULK||!peer||!header){fg_error_set(err,FG_ERR_ARGUMENT,"invalid fabric receive-any arguments");return FG_ERR_ARGUMENT;}struct pollfd descriptors[FG_RANK_COUNT-1u];uint32_t ranks[FG_RANK_COUNT-1u],count=0;for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)if(rank!=f->rank){descriptors[count]=(struct pollfd){.fd=f->peer[rank][cls].fd,.events=POLLIN};ranks[count++]=rank;}if(timing){memset(timing,0,sizeof(*timing));timing->poll_start_ns=fabric_ns();}for(;;){int ready=poll(descriptors,count,-1);if(ready<0&&errno==EINTR)continue;if(ready<0){fg_error_set(err,FG_ERR_IO,"poll fabric: %s",strerror(errno));return FG_ERR_IO;}uint32_t ready_mask=0;for(uint32_t i=0;i<count;i++)if(descriptors[i].revents&(POLLIN|POLLERR|POLLHUP))ready_mask|=1u<<ranks[i];if(timing){timing->ready_ns=fabric_ns();timing->ready_mask=ready_mask;}for(uint32_t i=0;i<count;i++)if(ready_mask&(1u<<ranks[i])){*peer=ranks[i];return fg_fabric_recv_timed(f,*peer,cls,header,payload,capacity,bytes,timing,err);}}
}
fg_status fg_fabric_recv_any(fg_fabric *f,fg_fabric_class cls,uint32_t *peer,fg_frame_header *header,void *payload,uint32_t capacity,uint32_t *bytes,fg_error *err){return fg_fabric_recv_any_timed(f,cls,peer,header,payload,capacity,bytes,NULL,err);}
fg_status fg_fabric_wait_ready(fg_fabric *f,uint32_t class_mask,uint32_t *peer,fg_fabric_class *ready_class,fg_error *err){if(!f||!(class_mask&3u)||(class_mask&~3u)||!peer||!ready_class){fg_error_set(err,FG_ERR_ARGUMENT,"invalid fabric readiness wait");return FG_ERR_ARGUMENT;}struct pollfd descriptors[(FG_RANK_COUNT-1u)*2u];uint32_t ranks[(FG_RANK_COUNT-1u)*2u];fg_fabric_class classes[(FG_RANK_COUNT-1u)*2u];uint32_t count=0;for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)if(rank!=f->rank)for(uint32_t cls=0;cls<2u;cls++)if(class_mask&(1u<<cls)){descriptors[count]=(struct pollfd){.fd=f->peer[rank][cls].fd,.events=POLLIN};ranks[count]=rank;classes[count]=(fg_fabric_class)cls;count++;}for(;;){int ready=poll(descriptors,count,-1);if(ready<0&&errno==EINTR)continue;if(ready<0){fg_error_set(err,FG_ERR_IO,"poll fabric readiness: %s",strerror(errno));return FG_ERR_IO;}for(uint32_t i=0;i<count;i++)if(descriptors[i].revents&(POLLIN|POLLERR|POLLHUP)){*peer=ranks[i];*ready_class=classes[i];return FG_OK;}}}
fg_status fg_fabric_prep_header_recv(fg_fabric *f,uint32_t peer,fg_fabric_class cls,fg_frame_header *header,uint64_t tag,fg_error *err){
    if(!f||peer>=FG_RANK_COUNT||peer==f->rank||cls>FG_FABRIC_BULK||!header){fg_error_set(err,FG_ERR_ARGUMENT,"invalid async header recv");return FG_ERR_ARGUMENT;}
    return fg_uring_prep_recv(f->ring,f->peer[peer][cls].fixed_slot,header,sizeof(*header),tag,err);
}
fg_status fg_fabric_prep_payload_recv(fg_fabric *f,uint32_t peer,fg_fabric_class cls,void *payload,uint32_t bytes,uint64_t tag,fg_error *err){
    if(!f||peer>=FG_RANK_COUNT||peer==f->rank||cls>FG_FABRIC_BULK||!payload||!bytes){fg_error_set(err,FG_ERR_ARGUMENT,"invalid async payload recv");return FG_ERR_ARGUMENT;}
    return fg_uring_prep_recv(f->ring,f->peer[peer][cls].fixed_slot,payload,bytes,tag,err);
}
fg_status fg_fabric_io_flush(fg_fabric *f,uint32_t count,fg_error *err){
    if(!f){fg_error_set(err,FG_ERR_ARGUMENT,"invalid fabric flush");return FG_ERR_ARGUMENT;}
    return fg_uring_flush(f->ring,count,err);
}
fg_status fg_fabric_io_reap(fg_fabric *f,uint32_t min_count,fg_uring_cqe *out,uint32_t capacity,uint32_t *completed,fg_error *err){
    if(!f){fg_error_set(err,FG_ERR_ARGUMENT,"invalid fabric reap");return FG_ERR_ARGUMENT;}
    return fg_uring_reap(f->ring,min_count,out,capacity,completed,err);
}
