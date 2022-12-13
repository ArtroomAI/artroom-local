import React from 'react'
import ReactDOM from 'react-dom/client'
import {HashRouter } from "react-router-dom";
import { ChakraProvider, ColorModeScript} from '@chakra-ui/react'
import './index.css'
import App from './components/App'
import indexTheme from './themes/indexTheme';
import { extendTheme } from '@chakra-ui/react'
const theme = extendTheme(indexTheme);
 
ReactDOM.createRoot(document.getElementById('root')).render(
  <HashRouter hashType="noslash">
    <ColorModeScript/>
      <ChakraProvider theme={theme}>
        <App />
    </ChakraProvider>
  </HashRouter>
);