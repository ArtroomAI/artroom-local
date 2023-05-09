import React, { useState } from 'react';
import { Flex, Spacer, Grid, GridItem } from '@chakra-ui/react';
import { Routes, Route } from 'react-router-dom';
import PromptGuide from './PromptGuide';
import Body from './Body';
import Sidebar from './Sidebar';
import Settings from './Settings';
import Paint from './Paint';
import Queue from './Queue';
import SDSettings from './SDSettings/SDSettings';
import ImageViewer from './ImageViewer';

import { ModelMerger } from './ModelMerger';
import { UpdateProgressBar } from './UpdateProgressBar';
import ImageEditor from './ImageEditor';
import { QueueManager } from '../QueueManager';

import { AppSocket } from './AppSocket';

export default function App () {
    const [navSize, setNavSize] = useState<'small' | 'large'>('small');

    return (
        <>
            <Grid
                fontWeight="bold"
                gap="1"
                gridTemplateColumns={
                    navSize === 'large' ? "300px 1fr 250px" : "125px 1fr 250px"
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
                    <Sidebar navSize={navSize} setNavSize={setNavSize}  />
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
                                    <SDSettings tab='default' />
                                </>}
                                path="/" />

                            <Route
                                element={<>
                                    <Paint />

                                    <Spacer />

                                    <SDSettings tab='paint' />
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
            <AppSocket />
        </>
    );
}
