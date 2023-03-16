import React, { useCallback, useContext, useEffect, useRef } from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
import { addToQueueState, queuePausedState, queueState } from './atoms/atoms';
import { artroomPathState } from './SettingsManager';
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
    const queuedItemRef = useRef(null);
    const artroomPath = useRecoilValue(artroomPathState);
    
    const emit = useCallback((remove: boolean = true) => {
        if(remove) {
            const __queue = [...queue];
            __queue.shift();
            queuedItemRef.current = __queue[0] || null;
            setQueue(__queue);
        }
        if (queuedItemRef.current === null || isQueuePaused) {
            return;
        }
        socket.emit('generate', queuedItemRef.current);
    }, [socket, queue, isQueuePaused]);

    useEffect(() => {
        queuedItemRef.current = queue[0] || null;
        emit(false);
    }, [queue, emit]);

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
            window.api.saveQueue(JSON.stringify(queue), artroomPath);
        }
        window.addEventListener('beforeunload', saveQueue);
        return () => {
            window.removeEventListener('beforeunload', saveQueue);
        }
    }, [queue, artroomPath]);

    useEffect(() => {
        window.api.readQueue(artroomPath).then(queue => {
            const __queue = parseQueue(queue);
            if(__queue.length) {
                setQueuePaused(true);
            }
            setQueue(__queue);
        });
    }, [artroomPath]);

    return <></>;
}