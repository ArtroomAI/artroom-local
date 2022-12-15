import { useEffect, useState } from 'react';
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
    Icon
} from '@chakra-ui/react';
import { Routes, Route } from 'react-router-dom';
import PromptGuide from './PromptGuide';
import Body from './Body';
import Sidebar from './Sidebar';
import Settings from './Settings';
import Upscale from './Upscale';
import Paint from './Paint';
// Import InvokeAI from './InvokeAI';
import Queue from './Queue';
import SDSettings from './SDSettings';
import ImageViewer from './ImageViewer';
import EquilibriumAI from './EquilibriumAI';
import ProfileMenu from './ProfileMenu';
import LoginPage from './Login/LoginPage';
// Import Info from './Info';
import { IoMdCloud, IoMdCloudOutline } from 'react-icons/io';

function Main () {
    const { colorMode, toggleColorMode } = useColorMode();
    const [loggedIn, setLoggedIn] = useState(false);

    const [width, setWidth] = useRecoilState(atom.widthState);
    const [height, setHeight] = useRecoilState(atom.heightState);
    const [text_prompts, setTextPrompts] = useRecoilState(atom.textPromptsState);
    const [negative_prompts, setNegativePrompts] = useRecoilState(atom.negativePromptsState);
    const [batch_name, setBatchName] = useRecoilState(atom.batchNameState);
    const [steps, setSteps] = useRecoilState(atom.stepsState);
    const [aspect_ratio, setAspectRatio] = useRecoilState(atom.aspectRatioState);
    const [aspectRatioSelection, setAspectRatioSelection] = useRecoilState(atom.aspectRatioSelectionState);
    const [seed, setSeed] = useRecoilState(atom.seedState);
    const [use_random_seed, setUseRandomSeed] = useRecoilState(atom.useRandomSeedState);
    const [n_iter, setNIter] = useRecoilState(atom.nIterState);
    const [sampler, setSampler] = useRecoilState(atom.samplerState);
    const [cfg_scale, setCFGScale] = useRecoilState(atom.CFGScaleState);
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [use_cpu, setUseCPU] = useRecoilState(atom.useCPUState);
    const [image_save_path, setImageSavePath] = useRecoilState(atom.imageSavePathState);
    const [long_save_path, setLongSavePath] = useRecoilState(atom.longSavePathState);
    const [highres_fix, setHighresFix] = useRecoilState(atom.highresFixState);
    const [speed, setSpeed] = useRecoilState(atom.speedState);
    const [ckpt, setCkpt] = useRecoilState(atom.ckptState);
    const [ckpt_dir, setCkptDir] = useRecoilState(atom.ckptDirState);
    const [ckpts, setCkpts] = useRecoilState(atom.ckptsState);
    const [strength, setStrength] = useRecoilState(atom.strengthState);

    const [use_full_precision, setUseFullPrecision] = useRecoilState(atom.useFullPrecisionState);
    const [save_grid, setSaveGrid] = useRecoilState(atom.saveGridState);
    const [debug_mode, setDebugMode] = useRecoilState(atom.debugMode);
    const [delay, setDelay] = useRecoilState(atom.delayState);

    const { ToastContainer, toast } = createStandaloneToast();
    const [cloudMode, setCloudMode] = useRecoilState(atom.cloudModeState);

    const getCkpts = (event) => {
        window.getCkpts(ckpt_dir).then((result) => {
            setCkpts(result);
        });
    };

    useEffect(
        () => {
            getCkpts();
        },
        [ckpt_dir]
    );

    useEffect(
        () => {
            window.getSettings().then((result) => {
                const settings = JSON.parse(result);
                console.log(settings);
                setTextPrompts(settings.text_prompts);
                setNegativePrompts(settings.negative_prompts);
                setBatchName(settings.batch_name);
                setSteps(settings.steps);
                setAspectRatio(settings.aspect_ratio);
                setWidth(settings.width);
                setHeight(settings.height);
                setSeed(settings.seed);
                setUseRandomSeed(settings.use_random_seed);
                setInitImage(settings.init_image);
                setStrength(settings.strength);
                setCFGScale(settings.cfg_scale);
                setNIter(settings.n_iter);
                setSampler(settings.sampler);
                setImageSavePath(settings.image_save_path);
                setLongSavePath(settings.long_save_path);
                setHighresFix(settings.highres_fix);
                setCkpt(settings.ckpt);
                setCkptDir(settings.ckpt_dir);
                setUseCPU(settings.use_cpu);
                setSpeed(settings.speed);
                setDebugMode(settings.debug_mode);
                setUseFullPrecision(settings.use_full_precision);
                setDelay(settings.delay);
                setSaveGrid(settings.save_grid);

                window.runPyTests().then((result) => {

                    if (result === 'success\r\n') {
                        toast({
                            id: 'testing',
                            title: 'All Artroom paths & dependencies successfully found!',
                            status: 'success',
                            position: 'top',
                            duration: 5000,
                            isClosable: false
                        });
                    } else if (result.length > 0) {
                        toast({
                            id: 'testing',
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

    useEffect(
        () => {
            if (width > 0) {
                let newHeight = height;
                if (aspectRatioSelection !== 'Init Image' && aspectRatioSelection !== 'None') {
                    try {
                        const values = aspect_ratio.split(':');
                        const widthRatio = parseFloat(values[0]);
                        const heightRatio = parseFloat(values[1]);
                        if (!isNaN(widthRatio) && !isNaN(heightRatio)) {
                            newHeight = Math.min(
                                1920,
                                Math.floor(width * heightRatio / widthRatio / 64) * 64
                            );
                        }
                    } catch {

                    }
                    setHeight(newHeight);
                }
            }
        },
        [width, aspect_ratio]
    );

    return (
        <Grid
            fontWeight="bold"
            gap="1"
            gridTemplateColumns="0px 1fr 300px"
            gridTemplateRows="43px 1fr 30px"
            h="200px"
            templateAreas={`"nav null header"
                          "nav main main"
                          "nav main main"`}
        >
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
                                    value={cloudMode}
                                />
                            </VStack>
                        </HStack>
                        : <LoginPage setLoggedIn={setLoggedIn}>
                            Login
                        </LoginPage>
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
                            exact
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
            <Main />
        </RecoilRoot>
    );
}
export default App;
