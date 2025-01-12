import React from 'react';

import { FluentProvider, webDarkTheme } from '@fluentui/react-components';
import { createRoot } from 'react-dom/client';

import App from './App';

import './index.scss';

const root = createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <FluentProvider theme={webDarkTheme} >
      <App />
    </FluentProvider>
  </React.StrictMode>
);
