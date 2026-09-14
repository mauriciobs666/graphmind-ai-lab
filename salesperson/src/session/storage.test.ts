import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  SESSION_STORAGE_KEYS,
  clearParticipantSession,
  clearPresenterSession,
  loadParticipantSession,
  loadPresenterSession,
  participantAuthHeader,
  presenterAuthHeader,
  saveParticipantSession,
  savePresenterSession,
} from './storage';

const participant = {
  participantId: 'p-abc123',
  token: 'tok-1',
  displayName: 'Ada',
  language: 'en',
};

const presenter = { token: 'ptok-1' };

beforeEach(() => {
  window.localStorage.clear();
});

describe('storage keys', () => {
  it('uses separate keys for the two credentials', () => {
    expect(SESSION_STORAGE_KEYS.participant).toBe('salesperson.participant');
    expect(SESSION_STORAGE_KEYS.presenter).toBe('salesperson.presenter');
    expect(SESSION_STORAGE_KEYS.participant).not.toBe(SESSION_STORAGE_KEYS.presenter);
  });
});

describe('participant session persistence', () => {
  it('round-trips through localStorage', () => {
    saveParticipantSession(participant);
    expect(loadParticipantSession()).toEqual(participant);
  });

  it('returns null when nothing is stored', () => {
    expect(loadParticipantSession()).toBeNull();
  });

  it('clearing the participant leaves no trace', () => {
    saveParticipantSession(participant);
    clearParticipantSession();
    expect(loadParticipantSession()).toBeNull();
  });

  it('never stores the join response welcome field', () => {
    saveParticipantSession(participant);
    const raw = window.localStorage.getItem(SESSION_STORAGE_KEYS.participant);
    expect(raw).not.toContain('welcome');
  });
});

describe('presenter session persistence', () => {
  it('round-trips through localStorage', () => {
    savePresenterSession(presenter);
    expect(loadPresenterSession()).toEqual(presenter);
  });

  it('clearing the presenter leaves no trace', () => {
    savePresenterSession(presenter);
    clearPresenterSession();
    expect(loadPresenterSession()).toBeNull();
  });
});

describe('clearing one credential leaves the other alone', () => {
  it('clearing the participant does not touch the presenter credential', () => {
    saveParticipantSession(participant);
    savePresenterSession(presenter);
    clearParticipantSession();
    expect(loadParticipantSession()).toBeNull();
    expect(loadPresenterSession()).toEqual(presenter);
  });

  it('clearing the presenter does not touch the participant credential', () => {
    saveParticipantSession(participant);
    savePresenterSession(presenter);
    clearPresenterSession();
    expect(loadPresenterSession()).toBeNull();
    expect(loadParticipantSession()).toEqual(participant);
  });
});

describe('auth headers', () => {
  it('builds the participant header as Bearer <participantId>.<token>', () => {
    expect(participantAuthHeader(participant)).toBe('Bearer p-abc123.tok-1');
  });

  it('builds the presenter header as Bearer presenter.<token>', () => {
    expect(presenterAuthHeader(presenter)).toBe('Bearer presenter.ptok-1');
  });
});

describe('a broken localStorage degrades instead of throwing', () => {
  it('save/load/clear never throw when localStorage.getItem/setItem/removeItem throw', () => {
    const getItem = vi.spyOn(window.localStorage.__proto__, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    const setItem = vi.spyOn(window.localStorage.__proto__, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    const removeItem = vi
      .spyOn(window.localStorage.__proto__, 'removeItem')
      .mockImplementation(() => {
        throw new Error('blocked');
      });

    expect(() => saveParticipantSession(participant)).not.toThrow();
    expect(() => loadParticipantSession()).not.toThrow();
    expect(() => clearParticipantSession()).not.toThrow();
    expect(loadParticipantSession()).toBeNull();

    getItem.mockRestore();
    setItem.mockRestore();
    removeItem.mockRestore();
  });
});
