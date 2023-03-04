import React, { useCallback, useContext, useEffect } from 'react';
import { useRecoilState } from 'recoil';
import { addToQueueState, queuePausedState, queueState } from './atoms/atoms';
import { SocketContext } from './socket';

const parseQueue = (queue: string | undefined): any[] => {
    if(!queue) { // undefined | "" case
        return [];
    }

    try {
        const __queue = JSON.parse(queue);
        if (Array.isArray(__queue)) {
            return __queue;
        }
        return 'queue' in __queue ? __queue.queue : [];
    } catch(err) {
        return []; // safe check if JSON is corrupted
    }
}

export const QueueManager = () => {
    const socket = useContext(SocketContext);
    const [queue, setQueue] = useRecoilState(queueState);
    const [isQueuePaused, setQueuePaused] = useRecoilState(queuePausedState);
    const [addToQueue, setAddToQueue] = useRecoilState(addToQueueState);
    
    const emit = useCallback(() => {
        const __queue = [...queue];
        __queue.shift();
        setQueue(__queue);
        if(!isQueuePaused && __queue[0]) {
            socket.emit('generate', __queue[0]);
        }
    }, [socket, queue, isQueuePaused]);

    useEffect(() => {
        if(!isQueuePaused && queue[0]) {
            socket.emit('generate', queue[0]);
        }
    }, [socket, queue, isQueuePaused]);

    useEffect(() => {
        if(addToQueue) {
            setAddToQueue(false);
            setQueuePaused(false);
        }
    }, [addToQueue]);

    useEffect(() => {
        socket.on('job_done', emit);

        return () => {
            socket.off('job_done', emit);
        };
    }, [socket, emit]);

    useEffect(() => {
        const saveQueue = () => {
            window.api.saveQueue(JSON.stringify(queue));
        }
        window.addEventListener('beforeunload', saveQueue);
        return () => {
            window.removeEventListener('beforeunload', saveQueue);
        }
    }, [queue]);

    useEffect(() => {
        window.api.readQueue().then(queue => {
            const __queue = parseQueue(queue);
            if(__queue.length) {
                setQueuePaused(true);
            }
            setQueue(__queue);
        });
    }, []);

    return <></>;
}