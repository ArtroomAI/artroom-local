import React, { useCallback, useContext, useEffect, useState } from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Flex,
    Spacer,
    useToast,
    Grid,
    GridItem,
    VStack,
    HStack,
    Switch,
    Icon,
    Button
} from '@chakra-ui/react';
import { Routes, Route } from 'react-router-dom';
import { useInterval } from './Reusable/useInterval/useInterval';
import PromptGuide from './PromptGuide';
import Body from './Body';
import Sidebar from './Sidebar';
import Settings from './Settings';
import Paint from './Paint';
import Queue from './Queue';
import SDSettings from './SDSettings/SDSettings';
import ImageViewer from './ImageViewer';
import EquilibriumAI from './EquilibriumAI';
import ProfileMenu from './ProfileMenu';
import LoginPage from './Modals/Login/LoginPage';
import ProtectedReqManager from '../helpers/ProtectedReqManager';

import { IoMdCloud, IoMdCloudOutline } from 'react-icons/io';
import { ModelMerger } from './ModelMerger';
import path from 'path';
import { SocketContext } from '../socket';
import { ImageState } from '../atoms/atoms.types';
import { UpdateProgressBar } from './UpdateProgressBar';
import ImageEditor from './ImageEditor';
import { QueueManager } from '../QueueManager';

import { batchNameState, imageSavePathState } from '../SettingsManager';

export default function App () {
    // Connect to the server 
    const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL;
    const [loggedIn, setLoggedIn] = useState(false);

    const image_save_path = useRecoilValue(imageSavePathState);
    const batch_name = useRecoilValue(batchNameState);

    const toast = useToast({});
    const [cloudMode, setCloudMode] = useRecoilState(atom.cloudModeState);
    const setShard = useSetRecoilState(atom.shardState);
    const navSize = useRecoilValue(atom.navSizeState);
    const [cloudRunning, setCloudRunning] = useRecoilState(atom.cloudRunningState);
    const [latestImages, setLatestImages] = useRecoilState(atom.latestImageState);
    const setMainImage = useSetRecoilState(atom.mainImageState);
    const [showLoginModal, setShowLoginModal] = useRecoilState(atom.showLoginModalState);
    const [controlnetPreview, setControlnetPreview] = useRecoilState(atom.controlnetPreviewState);
    const [removeBackgroundPreview, setRemoveBackgroundPreview] = useRecoilState(atom.removeBackgroundPreviewState);
    const socket = useContext(SocketContext);

    const handleGetImages = useCallback((data: ImageState) => {
        if(latestImages.length > 0 && latestImages[0].batch_id !== data.batch_id) {
            setLatestImages([data]);
        } else {
            setLatestImages([...latestImages, data]);
        }
        setMainImage(data);
    }, [latestImages, setLatestImages, setMainImage])

    const handleControlnetPreview = useCallback((data: {controlnetPreview: string}) => {
        setControlnetPreview(data.controlnetPreview);
    }, [controlnetPreview])

    const handleRemoveBackgroundPreview = useCallback((data: {removeBackgroundPreview: string}) => {
        setRemoveBackgroundPreview(data.removeBackgroundPreview);
    }, [removeBackgroundPreview])
    
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
                    let newCloudImages = [];
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
                                    isClosable: true,
                                    containerStyle: {
                                        pointerEvents: 'none'
                                    }
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
                                    isClosable: true,
                                    containerStyle: {
                                        pointerEvents: 'none'
                                    }
                                });
                                //const timestamp = new Date().getTime();
                                console.log(imagePath);
                                let dataURL = job_list[i].images[j].url;
                                newCloudImages.push({"b64": dataURL})
                                window.api.saveFromDataURL(JSON.stringify({dataURL, imagePath}));
                            }
                        }
                    }
                    setLatestImages([...latestImages, ...newCloudImages]);
                    setMainImage(newCloudImages[newCloudImages.length-1]?.b64)
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

    return (
        <>
            <Grid
                fontWeight="bold"
                gap="1"
                gridTemplateColumns = {
                    navSize === 'large'
                    ? "300px 1fr 250px"
                    : "125px 1fr 250px"
                }
                gridTemplateRows="43px 1fr 30px"
                h="200px"
                templateAreas={`"nav null header"
                            "nav main main"
                            "nav main main"`}
            >
                {/* {showLoginModal && <LoginPage setLoggedIn={setLoggedIn}></LoginPage>}
                <GridItem
                    area="header"
                    justifySelf="center"
                    pt="3">
                    {
                        loggedIn
                            ? <HStack align="center">
                                <ProfileMenu setLoggedIn={setLoggedIn}/>

                                <VStack
                                    alignItems="center"
                                    spacing={0}>
                                    <Icon as={cloudMode
                                        ? IoMdCloud
                                        : IoMdCloudOutline} />

                                    <Switch
                                        colorScheme="teal"
                                        onChange={(e) => setCloudMode(e.target.checked)}
                                        checked={cloudMode}
                                    />
                                </VStack>
                            </HStack>
                            :  
                        <Button
                            aria-label="View"
                            variant="outline"
                            onClick={()=>{setShowLoginModal(true)}}
                            >
                            Login
                            {' '}
                        </Button>
                    }
                </GridItem> */}

                <GridItem
                    area="nav"
                    pl="2">
                    <Sidebar />
                </GridItem>

                <GridItem
                    area="main"
                    pl="2">
                    <Flex>
                        <Routes>
                            <Route
                                element={<>
                                    <Body />
                                    <Spacer />
                                    <SDSettings />
                                </>}
                                path="/" />

                            <Route
                                element={<>
                                    <Paint />

                                    <Spacer />

                                    <SDSettings />
                                </>}
                                path="/paint" />

                            <Route
                                element={<Queue />}
                                path="/queue" />

                            <Route
                                element={<ImageEditor />}
                                path="/image-editor" />

                            <Route
                                element={<ModelMerger />}
                                path="/merge" />

                            <Route
                                element={<ImageViewer />}
                                path="/imageviewer" />

                            <Route
                                element={<EquilibriumAI />}
                                path="/equilibriumai" />

                            <Route
                                element={<PromptGuide />}
                                path="/prompt-guide" />

                            <Route
                                element={<Settings />}
                                path="/settings" />
                        </Routes>
                    </Flex>
                </GridItem>
            </Grid>
            <UpdateProgressBar />
            <QueueManager />
        </>
    );
}
