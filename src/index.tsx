import React from 'react';
import ReactDOM from 'react-dom/client';
import { HashRouter } from 'react-router-dom';
import { ChakraProvider, ColorModeScript, extendTheme } from '@chakra-ui/react';
import './index.css';
// import './components/UnifiedCanvas/styles/index.scss'
import App from './components/App';
import indexTheme from './themes/indexTheme';

const theme = extendTheme(indexTheme);

ReactDOM.createRoot(document.getElementById('root')).render(<React.StrictMode>
    <HashRouter>
        <ColorModeScript />
        <ChakraProvider theme={theme}>
            <App />
        </ChakraProvider>
    </HashRouter>
</React.StrictMode>);
