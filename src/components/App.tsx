import React, { useEffect, useState } from 'react';
import { RecoilRoot, useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Flex,
    Spacer,
    createStandaloneToast,
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

import ArtroomIcon from '../images/icon.ico';
import MinimizeIcon from '../images/minimize.png';
import MaximizeIcon from '../images/maximize.png';
import CloseIcon from '../images/close.png';

const AppTopBar = () => {
    return (
        <div id="menu-bar">
            <div className="left" role="menu">
                <img src={ArtroomIcon} width="30px"/>
                <h3 id="artroom-head">ArtroomAI</h3>
            </div>
            <div className="right">
                <button className="menubar-btn" id="minimize-btn" onClick={window.api.minimizeWindow}><img src={MinimizeIcon} width="20px"/></button>
                <button className="menubar-btn" id="max-unmax-btn" onClick={window.api.maxUnmaxWindow}><img src={MaximizeIcon} width="20px"/></button>
                <button className="menubar-btn" id="close-btn" onClick={window.api.closeWindow}><img src={CloseIcon} width="20px"/></button>
            </div>
        </div>
    );
}

function Main () {
    // Connect to the server

    const { colorMode, toggleColorMode } = useColorMode();
    const [loggedIn, setLoggedIn] = useState(false);

    const [imageSettings, setImageSettings] = useRecoilState(atom.imageSettingsState)

    const [long_save_path, setLongSavePath] = useRecoilState(atom.longSavePathState);
    const [highres_fix, setHighresFix] = useRecoilState(atom.highresFixState);
    const [debug_mode, setDebugMode] = useRecoilState(atom.debugMode);
    const [delay, setDelay] = useRecoilState(atom.delayState);

    const { ToastContainer, toast } = createStandaloneToast();
    const [cloudMode, setCloudMode] = useRecoilState(atom.cloudModeState);
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);

    const [showLoginModal, setShowLoginModal] = useRecoilState(atom.showLoginModalState);

    //make sure cloudmode is off, while not signed in
    if (!loggedIn) {
        setCloudMode(false);
    }

    ProtectedReqManager.setCloudMode = setCloudMode;
    ProtectedReqManager.setLoggedIn = setLoggedIn;
    ProtectedReqManager.toast = toast;

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
                            <ProfileMenu setLoggedIn={setLoggedIn} />

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


function App () {
    return (
        <RecoilRoot>
            <AppTopBar/>
            <Main />
        </RecoilRoot>
    );
}
export default App;
