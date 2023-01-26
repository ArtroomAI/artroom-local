import React, { useCallback, useContext, useEffect, useState } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Flex,
    Spacer,
    useToast,
    Grid,
    GridItem,
    useColorMode,
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
import Upscale from './Upscale';
import Paint from './Paint';
import Queue from './Queue';
import SDSettings from './SDSettings';
import ImageViewer from './ImageViewer';
import EquilibriumAI from './EquilibriumAI';
import ProfileMenu from './ProfileMenu';
import LoginPage from './Modals/Login/LoginPage';
import ProtectedReqManager from '../helpers/ProtectedReqManager';

import { IoMdCloud, IoMdCloudOutline } from 'react-icons/io';
import { ModelMerger } from './ModelMerger';
import path from 'path';
import { Console } from './Console';
import { SocketContext } from '../socket';
import { ImageState } from '../atoms/atoms.types';

export default function App () {
    // Connect to the server
    const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL;
    const { colorMode, toggleColorMode } = useColorMode();
    const [loggedIn, setLoggedIn] = useState(false);

    const [imageSettings, setImageSettings] = useRecoilState(atom.imageSettingsState)

    const [long_save_path, setLongSavePath] = useRecoilState(atom.longSavePathState);
    const [highres_fix, setHighresFix] = useRecoilState(atom.highresFixState);
    const [debug_mode, setDebugMode] = useRecoilState(atom.debugMode);
    const [delay, setDelay] = useRecoilState(atom.delayState);

    const toast = useToast({});
    const [cloudMode, setCloudMode] = useRecoilState(atom.cloudModeState);
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
    const [cloudRunning, setCloudRunning] = useRecoilState(atom.cloudRunningState);
    const [latestImages, setLatestImages] = useRecoilState(atom.latestImageState);
    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);
    const [showLoginModal, setShowLoginModal] = useRecoilState(atom.showLoginModalState);

    const socket = useContext(SocketContext);

    const handleGetImages = useCallback((data: ImageState) => {
        if(latestImages.length > 0 && latestImages[0].batch_id !== data.batch_id) {
            setLatestImages([data]);
        } else {
            setLatestImages([...latestImages, data]);
        }
        setMainImage(data);
    }, [latestImages, setLatestImages, setMainImage])

    useEffect(() => {
        socket.on('get_images', handleGetImages);

        return () => {
          socket.off('get_images', handleGetImages);
        };
    }, [socket, handleGetImages]);

    //make sure cloudmode is off, while not signed in
    if (!loggedIn) {
        setCloudMode(false);
    }

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
                    let job_list = response.data.jobs;
                    let text = "";
                    let pending_cnt = 0;
                    let newCloudImages = [];
                    for (let i = 0; i < job_list.length; i++) {
                        for (let j = 0; j < job_list[i].images.length; j++) {
                            if (job_list[i].images[j].status == 'PENDING') {
                                pending_cnt = pending_cnt + 1;
                            } else if (job_list[i].images[j].status == 'SUCCESS') {
                                //text = text + "job_" + job_list[i].id.slice(0, 5) + 'img_' + job_list[i].images[j].id + '\n';
                                let img_name = job_list[i].id + '_' + job_list[i].images[j].id;
                                const imagePath = path.join(imageSettings.image_save_path, imageSettings.batch_name, img_name + "_cloud.jpg");
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



    useEffect(
        () => {
            window.api.getSettings().then((result) => {
                const settings = JSON.parse(result);

                const imageSettingsData = {
                    text_prompts: settings.text_prompts,
                    negative_prompts: settings.negative_prompts,
                    batch_name: settings.batch_name,
                    n_iter: settings.n_iter,
                    steps: settings.steps,
                    seed: settings.seed,
                    strength: settings.strength,
                    cfg_scale: settings.cfg_scale,
                    sampler: settings.sampler,
                    width: settings.width,
                    height: settings.height,
                    aspect_ratio: settings.aspect_ratio,
                    ckpt: settings.ckpt,
                    speed: settings.speed,
                    save_grid: settings.save_grid,
                    use_random_seed: settings.use_random_seed,
                    init_image: settings.init_image,
                    mask_image: '',
                    invert: false,
                    image_save_path: settings.image_save_path,
                    ckpt_dir: settings.ckpt_dir,
                    vae: settings.vae,
                }
                setImageSettings(imageSettingsData)

                setLongSavePath(settings.long_save_path);
                setHighresFix(settings.highres_fix);
                setDebugMode(settings.debug_mode);
                setDelay(settings.delay);

                window.api.runPyTests().then((result) => {
                    if (result === 'success\r\n') {
                        toast({
                            title: 'All Artroom paths & dependencies successfully found!',
                            status: 'success',
                            position: 'top',
                            duration: 2000,
                            isClosable: true
                        });
                    } else if (result.length > 0) {
                        toast({
                            title: result,
                            status: 'error',
                            position: 'top',
                            duration: 10000,
                            isClosable: true
                        });
                    }
                });
            });

            if (colorMode === 'light') {
                toggleColorMode();
            }
        },
        // load only once
        // eslint-disable-next-line react-hooks/exhaustive-deps
        []
    );

    return (
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
            {showLoginModal && <LoginPage setLoggedIn={setLoggedIn}></LoginPage>}
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
            </GridItem>

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
                            element={<Upscale />}
                            path="/upscale" />

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
    );
}
