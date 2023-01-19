import React from 'react';
import ReactDOM from 'react-dom/client';
import { HashRouter } from 'react-router-dom';
import { ChakraProvider, ColorModeScript, extendTheme } from '@chakra-ui/react';
import { io } from 'socket.io-client';
import './index.css';
import './components/UnifiedCanvas/styles/index.scss'
import { RecoilRoot } from 'recoil';
import App from './components/App';
import indexTheme from './themes/indexTheme';
import AppTopBar from './components/AppTopBar';

export const socket = io('http://localhost:5300');
export const SocketContext = React.createContext(socket);

const theme = extendTheme(indexTheme);

ReactDOM.createRoot(document.getElementById('root')).render(<React.StrictMode>
    <HashRouter>
        <ColorModeScript />
        <ChakraProvider theme={theme}>
            <RecoilRoot>
                <AppTopBar/>
                <SocketContext.Provider value={socket}>
                    <App />
                </SocketContext.Provider>
            </RecoilRoot>
        </ChakraProvider>
    </HashRouter>
</React.StrictMode>);
