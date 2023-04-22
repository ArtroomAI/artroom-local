import React, { useCallback, useContext, useEffect, useState } from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import { useToast, UseToastOptions } from '@chakra-ui/react';
import { useInterval } from './Reusable/useInterval/useInterval';
import ProtectedReqManager from '../helpers/ProtectedReqManager';

import path from 'path';
import { SocketContext } from '../socket';
import { ImageState } from '../atoms/atoms.types';

import { batchNameState, imageSavePathState } from '../SettingsManager';

export const AppSocket: React.FC = () => {
    // Connect to the server 
    const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL;
    const [loggedIn, setLoggedIn] = useState(false);

    const image_save_path = useRecoilValue(imageSavePathState);
    const batch_name = useRecoilValue(batchNameState);

    const toast = useToast({});
    const setCloudMode = useSetRecoilState(atom.cloudModeState);
    const setShard = useSetRecoilState(atom.shardState);

    const [cloudRunning, setCloudRunning] = useRecoilState(atom.cloudRunningState);
    const setLatestImages = useSetRecoilState(atom.latestImageState);
    const setMainImage = useSetRecoilState(atom.mainImageState);
    const [showLoginModal, setShowLoginModal] = useRecoilState(atom.showLoginModalState);
    const [controlnetPreview, setControlnetPreview] = useRecoilState(atom.controlnetPreviewState);
    const [removeBackgroundPreview, setRemoveBackgroundPreview] = useRecoilState(atom.removeBackgroundPreviewState);
    const socket = useContext(SocketContext);

    const handleGetImages = useCallback((data: ImageState) => {
        setLatestImages(latestImages => {
            if(latestImages.length > 0 && latestImages[0].batch_id !== data.batch_id) {
                return [data];
            }
            return [data, ...latestImages].slice(0, 500)
        });
        setMainImage(data);
    }, [setLatestImages, setMainImage])

    const handleControlnetPreview = useCallback((data: {controlnetPreview: string}) => {
        setControlnetPreview(data.controlnetPreview);
    }, [controlnetPreview])

    const handleRemoveBackgroundPreview = useCallback((data: {removeBackgroundPreview: string}) => {
        setRemoveBackgroundPreview(data.removeBackgroundPreview);
    }, [removeBackgroundPreview])

    useEffect(() => {
        const log = (options: UseToastOptions) => {
            if (options.id && toast.isActive(options.id)) {
                toast.update(options.id, options);
            } else {
                toast(options);
            }
        };
        socket.on('status', log); 
        return () => {
          socket.off('status', log); 
        };
    }, [socket, toast]);
    
    useEffect(() => {
        socket.on('get_images', handleGetImages); 

        return () => {
          socket.off('get_images', handleGetImages);
        };
    }, [socket, handleGetImages]);

    useEffect(() => {
        socket.on('get_controlnet_preview', handleControlnetPreview); 
        return () => {
            socket.off('get_controlnet_preview', handleControlnetPreview);
          };
    }, [socket, handleControlnetPreview]);

    useEffect(() => {
        socket.on('get_remove_background_preview', handleRemoveBackgroundPreview); 
        return () => {
            socket.off('get_remove_background_preview', handleRemoveBackgroundPreview);
          };
    }, [socket, handleRemoveBackgroundPreview]);

    //make sure cloudmode is off, while not signed in
    useEffect(() => {
        if (!loggedIn) {
            setCloudMode(false);
        }
    }, [loggedIn]);

    ProtectedReqManager.setCloudMode = setCloudMode;
    ProtectedReqManager.setLoggedIn = setLoggedIn;
    ProtectedReqManager.toast = toast;

    useInterval(
        () => {
            ProtectedReqManager.make_get_request(`${ARTROOM_URL}/image/get_status`).then((response: any) => {
                if (response.data.jobs.length == 0) {
                    toast({
                        title: 'Cloud jobs Complete!',
                        status: 'success',
                        position: 'top',
                        duration: 5000,
                        isClosable: true,
                        containerStyle: {
                            pointerEvents: 'none'
                        }
                    });
                    setCloudRunning(false);
                } else {
                    setShard(response.data.shards);
                    let job_list = response.data.jobs;
                    let text = "";
                    let pending_cnt = 0;
                    let newCloudImages: Partial<ImageState>[] = [];
                    for (let i = 0; i < job_list.length; i++) {
                        for (let j = 0; j < job_list[i].images.length; j++) {
                            if (job_list[i].images[j].status == 'PENDING') {
                                pending_cnt = pending_cnt + 1;
                            } else if (job_list[i].images[j].status == 'FAILED') {

                                let shard_refund = job_list[i].image_settings.shard_cost/job_list[i].image_settings.n_iter;
                                toast({
                                    title: 'Cloud Error Occurred, ' + shard_refund +' Shards Refunded to account',
                                    description: "Failure on Image id: " + job_list[i].images[j].id + " Job id: " + job_list[i].id,
                                    status: 'error',
                                    position: 'top',
                                    duration: 10000,
                                    isClosable: true
                                });
                            } else if (job_list[i].images[j].status == 'SUCCESS') {
                                //text = text + "job_" + job_list[i].id.slice(0, 5) + 'img_' + job_list[i].images[j].id + '\n';
                                let img_name = job_list[i].id + '_' + job_list[i].images[j].id;
                                const imagePath = path.join(image_save_path, batch_name, img_name + "_cloud.png");
                                toast({
                                    title: "Image completed: " + imagePath,
                                    status: 'info',
                                    position: 'top',
                                    duration: 5000,
                                    isClosable: true
                                });
                                //const timestamp = new Date().getTime();
                                console.log(imagePath);
                                let dataURL = job_list[i].images[j].url;
                                newCloudImages.push({"b64": dataURL})
                                window.api.saveFromDataURL(JSON.stringify({dataURL, imagePath}));
                            }
                        }
                    }
                    setLatestImages((latestImages) => [...latestImages, ...newCloudImages]);
                    setMainImage(newCloudImages[newCloudImages.length - 1])
                    toast({
                        title: 'Cloud jobs running!\n',
                        description: text + pending_cnt + " jobs pending",
                        status: 'info',
                        position: 'top',
                        duration: 5000,
                        isClosable: true,
                        containerStyle: {
                            pointerEvents: 'none'
                        }
                    });
                }

            }).catch((err: any) => {
                console.log(err);
            });
        },
        cloudRunning ? 5000 : null
    );

    return null;
}