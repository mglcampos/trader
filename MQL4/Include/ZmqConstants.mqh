// -*-mode: c; c-style: stroustrup; c-basic-offset: 4; coding: utf-8-dos -*-

#property copyright "Copyright 2013 OpenTrading"
#property link      "https://github.com/OpenTrading/"

/*
  Zmq constants, but I'm not sure which version.
*/

//+---------------------------------------------------------------------------------+
//| Types and Options variables. Copied from zmq.h
//+---------------------------------------------------------------------------------+

// Message Flags.
#define ZMQ_MAX_VSM_SIZE 30
/*  Message types. These integers may be stored in 'content' member of the    */
/*  message instead of regular pointer to the data.                           */
#define ZMQ_DELIMITER 31
#define ZMQ_VSM 32
/*  Message flags. ZMQ_MSG_SHARED is strictly speaking not a message flag     */
/*  (it has no equivalent in the wire format), however, making  it a flag     */
/*  allows us to pack the stucture tigher and thus improve performance.       */
#define ZMQ_MSG_MORE 1
#define ZMQ_MSG_SHARED 128
#define ZMQ_MSG_MASK 129


//  Socket options.
#define ZMQ_HWM 1
#define ZMQ_SWAP 3
#define ZMQ_AFFINITY 4
#define ZMQ_IDENTITY 5
#define ZMQ_SUBSCRIBE 6
#define ZMQ_UNSUBSCRIBE 7
#define ZMQ_RATE 8
#define ZMQ_RECOVERY_IVL 9
#define ZMQ_MCAST_LOOP 10
#define ZMQ_SNDBUF 11
#define ZMQ_RCVBUF 12
#define ZMQ_RCVMORE 13
#define ZMQ_FD 14
#define ZMQ_EVENTS 15
#define ZMQ_TYPE 16
#define ZMQ_LINGER 17
#define ZMQ_RECONNECT_IVL 18
#define ZMQ_BACKLOG 19
#define ZMQ_RECOVERY_IVL_MSEC 20   /*  opt. recovery time, reconcile in 3.x   */
#define ZMQ_RECONNECT_IVL_MAX 21



#define ZMQ_AFFINITY 4
#define ZMQ_BACKLOG 19
#define ZMQ_CONFLATE 54
#define ZMQ_CURVE 2
#define ZMQ_CURVE_PUBLICKEY 48
#define ZMQ_CURVE_SECRETKEY 49
#define ZMQ_CURVE_SERVER 47
#define ZMQ_CURVE_SERVERKEY 50
#define ZMQ_DEALER 5
#define ZMQ_DELAY_ATTACH_ON_CONNECT 39
#define ZMQ_DONTWAIT 1
#define ZMQ_EADDRINUSE 98
#define ZMQ_EADDRNOTAVAIL 99
#define ZMQ_EAFNOSUPPORT 97
#define ZMQ_EAGAIN 11
#define ZMQ_ECONNABORTED 103
#define ZMQ_ECONNREFUSED 111
#define ZMQ_ECONNRESET 104
#define ZMQ_EFAULT 14
#define ZMQ_EFSM 156384763
#define ZMQ_EHOSTUNREACH 113
#define ZMQ_EINPROGRESS 115
#define ZMQ_EINVAL 22
#define ZMQ_EMSGSIZE 90
#define ZMQ_EMTHREAD 156384766
#define ZMQ_ENETDOWN 100
#define ZMQ_ENETRESET 102
#define ZMQ_ENETUNREACH 101
#define ZMQ_ENOBUFS 105
#define ZMQ_ENOCOMPATPROTO 156384764
#define ZMQ_ENODEV 19
#define ZMQ_ENOMEM 12
#define ZMQ_ENOTCONN 107
#define ZMQ_ENOTSOCK 88
#define ZMQ_ENOTSUP 95
#define ZMQ_EPROTONOSUPPORT 93
#define ZMQ_ETERM 156384765
#define ZMQ_ETIMEDOUT 110
#define ZMQ_EVENTS 15
#define ZMQ_EVENT_ACCEPTED 32
#define ZMQ_EVENT_ACCEPT_FAILED 64
#define ZMQ_EVENT_ALL 2047
#define ZMQ_EVENT_BIND_FAILED 16
#define ZMQ_EVENT_CLOSED 128
#define ZMQ_EVENT_CLOSE_FAILED 256
#define ZMQ_EVENT_CONNECTED 1
#define ZMQ_EVENT_CONNECT_DELAYED 2
#define ZMQ_EVENT_CONNECT_RETRIED 4
#define ZMQ_EVENT_DISCONNECTED 512
#define ZMQ_EVENT_LISTENING 8
#define ZMQ_EVENT_MONITOR_STOPPED 1024
#define ZMQ_FAIL_UNROUTABLE 33
#define ZMQ_FD 14
#define ZMQ_FORWARDER 2
#define ZMQ_HAUSNUMERO 156384712
#define ZMQ_IDENTITY 5
#define ZMQ_IMMEDIATE 39
#define ZMQ_IO_THREADS 1
#define ZMQ_IO_THREADS_DFLT 1
#define ZMQ_IPC_PATH_MAX_LEN 107
#define ZMQ_IPV4ONLY 31
#define ZMQ_IPV6 42
#define ZMQ_LAST_ENDPOINT 32
#define ZMQ_LINGER 17
#define ZMQ_MAXMSGSIZE 22
#define ZMQ_MAX_SOCKETS 2
#define ZMQ_MAX_SOCKETS_DFLT 1023
#define ZMQ_MECHANISM 43
#define ZMQ_MORE 1
#define ZMQ_MULTICAST_HOPS 25
#define ZMQ_NOBLOCK 1
#define ZMQ_NULL 0
#define ZMQ_PAIR 0
#define ZMQ_PLAIN 1
#define ZMQ_PLAIN_PASSWORD 46
#define ZMQ_PLAIN_SERVER 44
#define ZMQ_PLAIN_USERNAME 45
#define ZMQ_POLLERR 4
#define ZMQ_POLLIN 1
#define ZMQ_POLLOUT 2
#define ZMQ_PROBE_ROUTER 51
#define ZMQ_PUB 1
#define ZMQ_PULL 7
#define ZMQ_PUSH 8
#define ZMQ_QUEUE 3
#define ZMQ_RATE 8
#define ZMQ_RCVBUF 12
#define ZMQ_RCVHWM 24
#define ZMQ_RCVMORE 13
#define ZMQ_RCVTIMEO 27
#define ZMQ_RECONNECT_IVL 18
#define ZMQ_RECONNECT_IVL_MAX 21
#define ZMQ_RECOVERY_IVL 9
#define ZMQ_REP 4
#define ZMQ_REQ 3
#define ZMQ_REQ_CORRELATE 52
#define ZMQ_REQ_RELAXED 53
#define ZMQ_ROUTER 6
#define ZMQ_ROUTER_BEHAVIOR 33
#define ZMQ_ROUTER_MANDATORY 33
#define ZMQ_ROUTER_RAW 41
#define ZMQ_SNDBUF 11
#define ZMQ_SNDHWM 23
#define ZMQ_SNDMORE 2
#define ZMQ_SNDTIMEO 28
#define ZMQ_STREAM 11
#define ZMQ_STREAMER 1
#define ZMQ_SUB 2
#define ZMQ_SUBSCRIBE 6
#define ZMQ_TCP_ACCEPT_FILTER 38
#define ZMQ_TCP_KEEPALIVE 34
#define ZMQ_TCP_KEEPALIVE_CNT 35
#define ZMQ_TCP_KEEPALIVE_IDLE 36
#define ZMQ_TCP_KEEPALIVE_INTVL 37
#define ZMQ_TYPE 16
#define ZMQ_UNSUBSCRIBE 7
#define ZMQ_VERSION 40003
#define ZMQ_VERSION_MAJOR 4
#define ZMQ_VERSION_MINOR 0
#define ZMQ_VERSION_PATCH 3
#define ZMQ_XPUB 9
#define ZMQ_XPUB_VERBOSE 40
#define ZMQ_XREP 6
#define ZMQ_XREQ 5
#define ZMQ_XSUB 10
#define ZMQ_ZAP_DOMAIN 55
