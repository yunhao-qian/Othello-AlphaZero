import { Button, makeStyles, tokens } from '@fluentui/react-components';
import { PlayRegular, StopRegular } from '@fluentui/react-icons';
import { useMemo, useReducer } from 'react';

import appReducer from './appReducer';
import PlayerPicker from './PlayerPicker';
import PlayerSettings from './PlayerSettings';

export default function App() {
  const styles = useStyles();
  const [appState, appDispatch] = useReducer(appReducer, {
    inGame: false,
    startEndButtonClicked: false,
    inAnimation: false,
    blackSettings: { type: 'human' },
    whiteSettings: { type: 'human' },
  });

  const agentWorker = useMemo(() => {
    return new Worker(new URL('./worker.ts', import.meta.url));
  }, []);

  function handleBlackSettingsChange(settings: PlayerSettings) {
    appDispatch({ type: 'blackSettingsChange', settings });
  }

  function handleWhiteSettingsChange(settings: PlayerSettings) {
    appDispatch({ type: 'whiteSettingsChange', settings });
  }

  function handleStartEndButtonClick() {
    appDispatch({ type: 'startEndButtonClick' });
  }

  function handleAnimationStart() {
    appDispatch({ type: 'animationStart' });
  };

  function handleAnimationEnd() {
    appDispatch({ type: 'animationEnd' });
  }

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <Button
          onClick={handleStartEndButtonClick}
          icon={appState.inGame ? <StopRegular /> : <PlayRegular />}
          appearance='primary'
        >
          <span className={styles.startEndButtonText}>
            {appState.inGame ? 'End Game' : 'Start Game'}
          </span>
        </Button>
        <div className={styles.playerPickerContainer}>
          <PlayerPicker
            label='Black'
            disabled={appState.inGame}
            settings={appState.blackSettings}
            onChange={handleBlackSettingsChange}
          />
          <PlayerPicker
            label='White'
            disabled={appState.inGame}
            settings={appState.whiteSettings}
            onChange={handleWhiteSettingsChange}
          />
        </div>
      </div>
    </div>
  );
}

const useStyles = makeStyles({
  root: {
    height: '100%',
    margin: 0,
    padding: 0,

    display: 'flex',
    flexDirection: 'column',
    alignItems: 'stretch',
  },
  header: {
    margin: 0,
    padding: `${tokens.spacingVerticalL} ${tokens.spacingHorizontalL} 0`,

    display: 'flex',
    alignItems: 'end',
    justifyContent: 'center',
    flexWrap: 'wrap',
    columnGap: tokens.spacingHorizontalL,
    rowGap: tokens.spacingVerticalL,
  },
  playerPickerContainer: {
    margin: 0,
    padding: 0,

    display: 'flex',
    justifyContent: 'center',
    flexWrap: 'wrap',
    columnGap: tokens.spacingHorizontalL,
    rowGap: tokens.spacingVerticalL,
  },
  startEndButtonText: {
    width: '6em',
    textAlign: 'end',
  },
});
