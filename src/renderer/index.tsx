import React from 'react';
import ReactDOM from 'react-dom/client';
import { HashRouter } from 'react-router-dom';
import { ChakraProvider, extendTheme, ColorModeScript } from '@chakra-ui/react';
import './index.css';
import './components/UnifiedCanvas/styles/index.scss'
import { RecoilRoot } from 'recoil';
import App from './components/App';
import indexTheme from './themes/indexTheme';
import AppTopBar from './components/AppTopBar';
import { SocketContext, socket } from './socket';
import { InstallerManager } from './InstallerManager';

const theme = extendTheme(indexTheme);

ReactDOM.createRoot(document.getElementById('root')!).render(<HashRouter>
    <ColorModeScript initialColorMode='dark' />    
    <ChakraProvider theme={theme}>
        <RecoilRoot>
            <AppTopBar/>
            <SocketContext.Provider value={socket}>
                <App />
            </SocketContext.Provider>
            <InstallerManager />
        </RecoilRoot>
    </ChakraProvider>
</HashRouter>);
