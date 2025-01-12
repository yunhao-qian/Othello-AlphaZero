import React from 'react';

import { FluentProvider, webDarkTheme } from '@fluentui/react-components';
import { createRoot } from 'react-dom/client';

const root = createRoot(document.getElementById('root'));

root.render(
  <FluentProvider theme={webDarkTheme} >
  </FluentProvider>,
);
