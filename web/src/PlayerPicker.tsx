import {
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  DialogTrigger,
  Label,
  Select,
  SelectOnChangeData,
  makeStyles,
  tokens,
  useId
} from '@fluentui/react-components';
import { SettingsRegular } from '@fluentui/react-icons';
import React, { useState } from 'react';

import AlphaZeroSettingsDialog from './AlphaZeroSettingsDialog';
import PlayerSettings from "./PlayerSettings";

export default function PlayerPicker({ label, disabled, settings, onChange }: PlayerPickerProps) {
  const styles = useStyles();
  const playerTypeSelectId = useId();
  const [modifiedSettings, setModifiedSettings] = useState<PlayerSettings | null>(settings);

  function handlePlayerTypeChange(
    event: React.ChangeEvent<HTMLSelectElement>, data: SelectOnChangeData
  ) {
    switch (PLAYER_TYPE_VALUES[PLAYER_TYPE_TEXTS.indexOf(data.value)]) {
      case 'human':
        onChange({ type: 'human' });
        break;
      case 'alphaZero':
        onChange({ type: 'alphaZero', numSimulations: 800, batchSize: 16 });
      default:
        console.error('Invalid player type:', data.value);
    }
  }

  let settingsDialog: React.ReactNode = null;
  if (!disabled) {
    switch (settings.type) {
      case 'alphaZero':
        settingsDialog = (
          <AlphaZeroSettingsDialog settings={settings} onChange={setModifiedSettings} />
        );
        break;
    }
  }

  function handleSettingsDialogCancel() {
    setModifiedSettings(settings);
  }

  function handleSettingsDialogSave() {
    if (modifiedSettings !== null) {
      onChange(modifiedSettings);
    }
  }

  const playerTypeText = PLAYER_TYPE_TEXTS[PLAYER_TYPE_VALUES.indexOf(settings.type)];

  return (
    <div className={styles.root}>
      <div>
        <Label htmlFor={playerTypeSelectId}>{`${label}:`}</Label>
        <Select
          id={playerTypeSelectId}
          disabled={disabled}
          value={playerTypeText}
          onChange={handlePlayerTypeChange}
        >
          {PLAYER_TYPE_TEXTS.map((text) => (
            <option key={text}>{text}</option>
          ))}
        </Select>
      </div>
      <Dialog modalType='alert'>
        <DialogTrigger disableButtonEnhancement>
          <Button disabled={settingsDialog === null} icon={<SettingsRegular />} />
        </DialogTrigger>
        <DialogSurface>
          <DialogBody>
            <DialogTitle>{`${label} Settings: ${playerTypeText}`}</DialogTitle>
            <DialogContent>
              {settingsDialog}
            </DialogContent>
            <DialogActions>
              <DialogTrigger disableButtonEnhancement>
                <Button appearance="secondary" onClick={handleSettingsDialogCancel}>Cancel</Button>
              </DialogTrigger>
              <DialogTrigger>
                <Button
                  appearance="primary"
                  disabled={modifiedSettings === null}
                  onClick={handleSettingsDialogSave}
                >
                  Save
                </Button>
              </DialogTrigger>
            </DialogActions>
          </DialogBody>
        </DialogSurface>
      </Dialog>
    </div>
  );
}

export type PlayerPickerProps = {
  label: string;
  disabled: boolean;
  settings: PlayerSettings;
  onChange: (settings: PlayerSettings) => void;
};

const useStyles = makeStyles({
  root: {
    display: 'flex',
    alignItems: 'end',
    gap: tokens.spacingHorizontalS,
  },
});

const PLAYER_TYPE_VALUES = ['human', 'alphaZero'];
const PLAYER_TYPE_TEXTS = ['Human', 'AlphaZero'];
